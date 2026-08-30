-- 002_query_traces.sql
-- Course-owned QUERY_TRACES table.
-- Run as dmuser against FREEPDB1.
--
-- Records each SQL query issued by the course scripts along with its latency
-- and success status.  Used by course_lab/vsql.py to mine query templates.
--
-- Idempotency: ORA-00955 is silently swallowed.

BEGIN
  EXECUTE IMMEDIATE '
    CREATE TABLE QUERY_TRACES (
      trace_id   VARCHAR2(64)  PRIMARY KEY,
      sql_text   CLOB          NOT NULL,
      latency_ms NUMBER,
      success    NUMBER(1)     DEFAULT 1,
      created_ts TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
    )
  ';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE != -955 THEN RAISE; END IF;
END;
/
