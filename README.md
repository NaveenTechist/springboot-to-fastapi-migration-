# Spring Boot → FastAPI Migration – LMST Backend

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)  
[![FastAPI](https://img.shields.io/badge/FastAPI-HighPerformance-green)](https://fastapi.tiangolo.com/)  
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue)](https://www.postgresql.org/)  
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 📌 Project Overview

This repository demonstrates a **real-world backend migration** from **Spring Boot (Java)** to **FastAPI (Python)** for the LMST system – a data-driven, scheduled, I/O-heavy backend application.

The migration evaluates whether **Python + FastAPI** can replace Spring Boot for workloads characterized by:

- Heavy database usage (PostgreSQL + stored procedures)  
- Large-scale data processing (millions of rows)  
- Scheduled document ingestion jobs  
- High concurrency with low-to-moderate CPU usage  
- Long-term maintainability and future AI/ML integration readiness  

> ⚠️ This is not a rewrite for trend-following purposes. The goal is **technical optimization** for LMST’s data pipeline workload.

---

## 🎯 Purpose of Migration

The migration focuses on:

1. Improving **data-processing flexibility**  
2. Enabling **async, non-blocking I/O**  
3. Simplifying **document → table ingestion pipelines**  
4. Supporting **scheduled automated jobs**  
5. Preparing the backend for **future AI/ML integrations**

---

## ⚙️ LMST Workload Characteristics

| Area | Nature |
|------|-------|
| API Traffic | I/O-bound |
| Database | PostgreSQL only |
| Data Volume | Large (millions of rows) |
| Business Logic | Mostly in DB (stored procedures) |
| Scheduling | Daily fixed-time jobs (e.g., 9 AM) |
| File Processing | Document → Table ingestion |
| CPU Usage | Low–moderate |
| Concurrency | High DB + file I/O |

> The workload is **I/O-heavy**, making FastAPI's async-first architecture a natural fit.

---

## 🏗️ Architecture Comparison

| Feature | Spring Boot (Java) | FastAPI (Python) |
|---------|-----------------|----------------|
| Controller | Controller | Router |
| Service | Service Layer | Service Layer |
| Repository | Repository Layer | Repository Layer |
| DTO | DTO Classes | Pydantic Schemas |
| Config | `application.yml` | `.env` / Python settings |
| Server | Tomcat | Uvicorn |
| Concurrency | Thread-based | Async / Event loop |
| Best For | CPU-heavy enterprise workloads | Data pipelines, async I/O workloads |

> LMST’s **daily data ingestion & async DB processing** workload aligns better with **FastAPI architecture**.

---


---

## ⚡ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI |
| Language | Python 3.12 |
| Server | Uvicorn |
| Validation | Pydantic |
| Database | PostgreSQL |
| Config Management | python-dotenv |
| Logging | Python Logging module |
| File Handling | shutil, gzip, pandas |
| Authentication | JWT / bcrypt |

---

## 🔹 Key Features

- ✅ **Async I/O** for DB & file operations  
- ✅ **Document ingestion pipeline**: `.gz`, `.xls`, `.xlsx` → CSV → PostgreSQL  
- ✅ **Background tasks** with `BackgroundTasks`  
- ✅ **JWT-based authentication**  
- ✅ **Dynamic DB stored procedure execution**  
- ✅ **Configurable via `.env` file**  
- ✅ **Cross-Origin Resource Sharing (CORS) support**  


