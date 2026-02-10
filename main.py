from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import shutil
import os
import psycopg2
import base64
import bcrypt
import jwt
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging
import io
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncpg
import logging

logger = logging.getLogger("reports")


load_dotenv()  # Load .env file



# DB Config
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
JWT_SECRET = os.getenv("JWT_SECRET", "default-secret-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60  # Token valid for 60 minutes

app = FastAPI()



CORS_ORIGINS = ["http://localhost:5000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,   # must be list of exact origins
    allow_credentials=True,       # only works with exact origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Authorization"],  
)


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

@app.get("/signin")
async def signin(request: Request, response: Response, code: str | None = None):
    try:
        # 1️⃣ Decode Basic Auth
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
        encoded = auth_header.split(" ")[1]
        decoded = base64.b64decode(encoded).decode("utf-8")
        username, password = decoded.split(":", 1)

        # 2️⃣ Fetch user from DB
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, employeeid, username, password, brname, branchno, city, role, status, mis, slbc, reconciliation, expiry_date 
            FROM userloginrj WHERE employeeid=%s
        """, (username,))
        user_row = cur.fetchone()
        cur.close()
        conn.close()

        if not user_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        print(user_row)

        # 3️⃣ Map tuple to dict
        columns = ["id", "employeeid", "username", "password", "brname", "branchno", "city", "role", "status", "mis", "slbc", "reconciliation", "expiry_date"]
        user = dict(zip(columns, user_row))
        print(columns)

        # 4️⃣ Verify password
        hashed_password = user["password"]
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode("utf-8")
        if not bcrypt.checkpw(password.encode("utf-8"), hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong password")

        # 5️⃣ Generate REAL JWT token
        now_ts = int(datetime.utcnow().timestamp())
        exp_ts = int((datetime.utcnow() + timedelta(hours=3)).timestamp())
        expire = datetime.utcnow() + timedelta(hours=3)
        payload = {
            "iss": "misreports",
            "sub": "JWT Token",
            "username": user["username"],
            "role": user["role"],
            "schema": "public",
            "iat": now_ts,
            "exp": exp_ts
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

        response.headers["Authorization"] = f"Bearer {token}"

        # 6️⃣ Return response
        return {
            "error": False,
            "status_code": 202,
            "token": token,
            "employeeid": str(user["employeeid"]),
            "username": user["username"],
            "brname": user["brname"],
            "branchno": user["branchno"],
            "city": user["city"],
            "role": user["role"],
            "status": user["status"],
            "mis": user["mis"],
            "slbc": user["slbc"],
            "reconciliation": user["reconciliation"],
            "expiry_date": str(user["expiry_date"]),
            "schema": "public",
            "code": code
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/paths")
def get_paths() -> dict:
    """Return loader path configuration from env (FROM_PATHS, TO_PATH)."""

    # FROM_PATHS
    from_paths_raw = os.environ.get("FROM_PATHS", "").strip()
    if not from_paths_raw:
        raise RuntimeError("Environment variable 'FROM_PATHS' is missing or empty.")

    from_paths = [p.strip() for p in from_paths_raw.split(",") if p.strip()]
    if not from_paths:
        raise RuntimeError("Environment variable 'FROM_PATHS' is empty after parsing.")

    from_path_str = ",".join(from_paths)  # no extra spaces

    # TO_PATH
    to_path_raw = os.environ.get("TO_PATH", "").strip()
    if not to_path_raw:
        raise RuntimeError("Environment variable 'TO_PATH' is missing or empty.")

    return {
        "fromPath": from_path_str,
        "toPath": to_path_raw
    }




class FileItem(BaseModel):
    key: str
    name: str
    display_name: str
    files: int
    file_type: str
    db: str

class FileRequest(BaseModel):
    fromPath: str
    toPath: str
    startDate: str
    endDate: str
    files: List[FileItem]

# --------------------
# Utilities
# --------------------
def delete_all_files_in_directory(path: str):
    if not os.path.exists(path):
        return
    for f in Path(path).glob("*"):
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)

def unzip_file(source: Path, dest: Path) -> Path:
    """Unzip .gz recursively"""
    current = source
    while current.suffix == ".gz":
        out_file = dest / current.stem
        with gzip.open(current, "rb") as f_in, open(out_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        current = out_file
    return current

def convert_xls_to_csv(file_path: Path, dest_path: Path):
    """Placeholder for XLS/XLSX → CSV logic (can use pandas or openpyxl)"""
    import pandas as pd
    df = pd.read_excel(file_path)
    df.to_csv(dest_path, index=False)

def send_status(msg_type: str, msg: str):
    """Simulate WebSocket message"""
    logging.info(f"[{msg_type}] {msg}")

# --------------------
# Database Processing
# --------------------

# # metadata.py
# FILE_METADATA = {
#     "MONTRIAL": {
#         "tableName": "mon_trial_daily",
#         "sequenceName": "mon_trial_daily_rno_seq",
#         "procedureName": "mon_trial_load",
#         "useFilePath": False
#     }
# }


# def get_dynamic_script(file_name: str, file_path: str) -> list[str]:
#     meta = FILE_METADATA.get(file_name.upper())
#     if not meta:
#         raise ValueError(f"No DB script for file: {file_name}")

#     proc = meta.get("procedureName")
#     use_path = meta.get("useFilePath", False)

#     if not proc:
#         raise ValueError(f"No procedure defined for file: {file_name}")

#     if use_path:
#         return [f"CALL {proc}('{file_path}')"]
#     else:
#         return [f"CALL {proc}()"]



def get_dynamic_script(file_name: str, file_path: str) -> list[str]:
    if file_name.upper() == "MONTRIAL":
        return [
            "TRUNCATE TABLE mon_trial_daily;",
            "CALL mon_trial_load();",
            """INSERT INTO MONTRIAL
               (BRCODE,CGL,CGLDESC,OPEN_BAL,DEBIT_BAL,CREDIT_BAL,NET_CHANGE,END_BAL,DATE_OF_REPORT)
               SELECT 
               BRCD::NUMERIC,
               CGL::NUMERIC,
               CGLDESC,OPEN_BAL,DEBIT_BAL,CREDIT_BAL,NET_CHANGE,END_BAL,REPORT_DT 
               FROM MON_TRIAL_DAILY
               WHERE END_BAL IS NOT NULL;""",
            "CALL mon_trial_drcr();"
        ]
    else:
        raise ValueError("No DB script for file")

def get_schema_for_branch(branch_code: str) -> str:
    raw = os.getenv("BRANCH_SCHEMA_MAP", "")
    if not raw:
        raise RuntimeError("BRANCH_SCHEMA_MAP not set in .env")

    mappings = dict(item.split(":") for item in raw.split(","))
    
    schema = mappings.get(branch_code)
    if not schema:
        raise RuntimeError(f"No schema mapped for branch {branch_code}")

    return schema




def run_db_script(file_path: str, file_name: str, db_conn_str: str):
    try:
        sqls = get_dynamic_script(file_name, file_path)

        conn = psycopg2.connect(db_conn_str)
        cur = conn.cursor()

        schema = os.getenv("BRANCH_SCHEMA", "public")
        cur.execute(f"SET search_path TO {schema}")

        for sql in sqls:

            # After TRUNCATE → do copy
            if "TRUNCATE TABLE mon_trial_daily" in sql:
                cur.execute(sql)

                # 🔥 CLEAN + COPY
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    cleaned = []
                    for line in f:
                        line = line.replace('\r', '').replace('\x00', '')
                        cleaned.append(line.rstrip())

                buffer = io.StringIO("\n".join(cleaned))

                cur.copy_expert(
                    "COPY mon_trial_daily(fulltext) FROM STDIN WITH (FORMAT text)",
                    buffer
                )

            else:
                cur.execute(sql)

        conn.commit()
        cur.close()
        conn.close()

        msg = f"✅ DB Processing completed for {file_name} ({schema})"
        send_status("success", msg)
        return msg

    except Exception as e:
        msg = f"❌ DB error {file_name}: {e}"
        send_status("error", msg)
        return msg


async def process_file(file: FileItem, request: FileRequest):
    matched_files = []
    start = datetime.strptime(request.startDate, "%Y-%m-%d").date()
    end = datetime.strptime(request.endDate, "%Y-%m-%d").date()
    src_dir = Path(request.fromPath)
    dest_dir = Path(request.toPath)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Here just simulate date folder filtering
    for folder in os.listdir(src_dir):
        folder_date = None
        try:
            folder_date = datetime.strptime(folder, "%Y%m%d").date()
        except:
            continue
        if not (start <= folder_date <= end):
            continue

        folder_path = src_dir / folder  #ERROR COMING FROM HERE....
        for f in folder_path.glob("*"):
            if file.file_type == "prefix" and f.name.lower().startswith(file.key.lower()):
                actual_file = f
                if f.suffix == ".gz":
                    actual_file = unzip_file(f, dest_dir)
                dest_file = dest_dir / actual_file.name
                if actual_file.suffix in [".xls", ".xlsx"]:
                    dest_file = dest_dir / f"{actual_file.stem}.csv"
                    convert_xls_to_csv(actual_file, dest_file)
                else:
                    shutil.copy(actual_file, dest_file)
                # Process DB (sync for simplicity, async can be added)
                result = run_db_script(str(dest_file), file.display_name, "dbname=nyx user=postgres password=123 host=localhost port=5433")
                matched_files.append(result)
                if file.files == 1:
                    break
    if not matched_files:
        msg = f"❌ No matching file for {file.display_name}"
        send_status("error", msg)
        matched_files.append(msg)
    return matched_files

# --------------------
# Endpoint
# --------------------
@app.post("/copy")
async def copy_endpoint(request: FileRequest, background_tasks: BackgroundTasks):
    if not request.files:
        raise HTTPException(status_code=400, detail="At least one file must be selected")
    delete_all_files_in_directory(request.toPath)

    results = await asyncio.gather(*(process_file(f, request) for f in request.files))

    # Dashboard refresh for special files
    if any(f.name.lower() == "montrial" for f in request.files):
        background_tasks.add_task(lambda: send_status("info", "Dashboard refreshed"))

    # Flatten results
    flat_results = [item for sublist in results for item in sublist]
    return flat_results



from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import psycopg2
import psycopg2.extras
import logging
import asyncpg

# =========================================================
# QUERY MAP GEN001
# =========================================================

QUERY_MAP = {
    "GEN001": """
        select
          ROW_NUMBER() OVER () AS serial_no,
          report_date,
          report_type,
          procedure_name,
          file_name,
          inserted_records,
          to_char(execution_time AT TIME ZONE 'Asia/Kolkata',
                  'DD Mon YYYY HH12:MI:SS AM') as execution_time,
          to_char(end_time AT TIME ZONE 'Asia/Kolkata',
                  'DD Mon YYYY HH12:MI:SS AM') as end_time,
          status,
          round(extract(epoch from (end_time - execution_time))::numeric, 2)
            as time_in_seconds,
          error_count,
          count(*) over() as total_rows
        from procedure_log
        where report_date between %s and %s
    """
}

# =========================================================
# REQUEST MODELS
# =========================================================

class Filter(BaseModel):
    id: str
    value: str


class ReportRequest(BaseModel):
    reportType: str
    branch: str
    fromDate: str
    toDate: str
    page: int = 1
    size: int = 10
    sortBy: Optional[str] = None
    sortDirection: Optional[str] = "ASC"
    filters: Optional[List[Filter]] = []


# =========================================================
# UTILS
# =========================================================

def format_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%d-%m-%Y")


def apply_filters(query: str, filters: List[Filter]) -> str:
    if not filters:
        return query

    clauses = []
    for f in filters:
        clauses.append(
            f"CAST({f.id} AS TEXT) LIKE '%{f.value}%'"
        )

    return f"select * from ({query}) t WHERE " + " AND ".join(clauses)


def add_pagination_sorting(
    query: str,
    page: int,
    size: int,
    sort_by: Optional[str],
    sort_dir: Optional[str]
) -> str:

    if sort_by:
        query += f" ORDER BY {sort_by} {sort_dir}"

    # HERE: size = -1 means no pagination ================================= changes if you want to
    if size != -1:
        offset = max((page - 1) * size, 0)
        query += f" LIMIT {size} OFFSET {offset}"

    return query


# =========================================================
# SINGLE /reports ENDPOINT
# =========================================================

@app.post("/reports")
def get_reports(request: ReportRequest) -> List[Dict[str, Any]]:
    try:
        # -------------------------------------------------
        # Branch validation (kept compatible)
        # -------------------------------------------------
        if request.branch == "":
            raise ValueError("Branch is empty")

        # HERE: Branch parsed but NOT used in GEN001 query
        branch = int(request.branch)

        # -------------------------------------------------
        # Get SQL
        # -------------------------------------------------
        base_query = QUERY_MAP.get(request.reportType)
        if not base_query:
            raise HTTPException(status_code=400, detail="Invalid report type")

        # -------------------------------------------------
        # Date formatting
        # -------------------------------------------------
        from_date = format_date(request.fromDate) if request.fromDate else None
        to_date = format_date(request.toDate)

        # -------------------------------------------------
        # Filters
        # -------------------------------------------------
        final_query = apply_filters(base_query, request.filters)

        # -------------------------------------------------
        # Pagination + Sorting
        # -------------------------------------------------
        final_query = add_pagination_sorting(
            final_query,
            request.page,
            request.size,
            request.sortBy,
            request.sortDirection
        )

        # -------------------------------------------------
        # DB EXECUTION
        # -------------------------------------------------
        # HERE: Reuse your existing DB config
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(final_query, (from_date, to_date))
                result = cur.fetchall()
                return result
        finally:
            conn.close()

    except psycopg2.errors.QueryCanceled:
        logger.error("Query timeout")
        raise HTTPException(
            status_code=408,
            detail="Request timed out. Please try again with different filters."
        )

    except psycopg2.errors.SyntaxError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Arguments")

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)

    