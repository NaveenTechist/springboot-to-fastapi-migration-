# Spring Boot → FastAPI Migration – LMST Backend (Report Generation Focus)

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)  
[![FastAPI](https://img.shields.io/badge/FastAPI-HighPerformance-green)](https://fastapi.tiangolo.com/)  
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue)](https://www.postgresql.org/)  
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 📌 Project Overview

This repository demonstrates a **backend migration** from **Spring Boot (Java)** to **FastAPI (Python)** for the LMST system, with a **focus on report generation performance**.

The goal is to evaluate whether Python + FastAPI can handle **high-volume database reports** efficiently while maintaining:

- Heavy PostgreSQL usage (millions of rows, stored procedures)  
- Scheduled and on-demand report generation  
- High I/O concurrency  
- Low-to-moderate CPU usage  
- Ease of maintenance and future AI/ML integration  

> ⚠️ This is a **performance-driven migration**, primarily to test and compare report generation times between Spring Boot and FastAPI.

---

## 🎯 Purpose of Migration (Report-Focused)

The migration is focused on:

1. Measuring **report generation performance**  
2. Supporting **async, non-blocking I/O** for large datasets  
3. Enabling **scheduled and on-demand report jobs**  
4. Simplifying **document → table ingestion pipelines**  
5. Preparing for future **AI/ML report analytics**

---

## ⚡ Report Generation Performance (Speed Test)

| Framework | Run Times (seconds) | Notes |
|-----------|------------------|-------|
| **Spring Boot** | 2.55, 1.59, 1.65, 2.18 | Consistent for `MONTRIAL` report |
| **FastAPI (Python)** | 1.59, 1.81, 3.49, 1.93 | Sometimes slightly higher due to async scheduling, but comparable overall |

> 🔹 Observations:  
> - FastAPI achieves **similar or better times** for I/O-heavy report workloads.  
> - Variability in FastAPI is due to **event loop scheduling and DB async calls**, but peak performance is still competitive.  
> - Spring Boot remains stable for **CPU-light, high-DB workloads**, but FastAPI offers **async concurrency** benefits for future scaling.

---

## ⚙️ LMST Workload Characteristics (Report Focus)

| Area | Nature |
|------|-------|
| API Traffic | I/O-bound |
| Database | PostgreSQL only |
| Data Volume | Large (millions of rows) |
| Business Logic | Mostly in DB (stored procedures for reports) |
| Scheduling | Daily and on-demand report generation |
| CPU Usage | Low–moderate |
| Concurrency | High DB + file I/O |

> The **report generation workload** is I/O-heavy, making FastAPI’s async-first architecture ideal for scaling multiple report requests.

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
| Best For | Stable, CPU-light report generation | Async report pipelines, I/O-heavy workloads |

> FastAPI is well-suited for **parallel report generation** and **high-concurrency DB access**.

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
| Reports | `MONTRIAL` |

---

## 🔹 Key Features (Report-Focused)

- ✅ **Async I/O** for DB and file operations  
- ✅ **Document ingestion pipeline**: `.gz`, `.xls`, `.xlsx` → CSV → PostgreSQL  
- ✅ **Dynamic stored procedure execution** for report generation  
- ✅ **Background tasks** with FastAPI `BackgroundTasks`  
- ✅ **JWT-based authentication for API access**  
- ✅ **Configurable via `.env` file**  
- ✅ **CORS support for dashboard clients**  
- ✅ **Performance logging** for each report run (execution time in seconds)  
