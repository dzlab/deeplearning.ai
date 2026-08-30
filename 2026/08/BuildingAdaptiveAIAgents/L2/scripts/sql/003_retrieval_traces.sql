-- 003_retrieval_traces.sql
-- Course-owned RETRIEVAL_TRACES table.
-- Run as dmuser against FREEPDB1.
--
-- Records retrieval events: what was queried, which candidates were surfaced by
-- OracleAgentMemory.search(), which were chosen/rejected, and the final
-- outcome.  Used by course_lab/embedding_adapt.py to mine positive/negative
-- pairs for projection-adapter training.
--
-- Idempotency: ORA-00955 is silently swallowed.

BEGIN
  EXECUTE IMMEDIATE '
    CREATE TABLE RETRIEVAL_TRACES (
      trace_id   VARCHAR2(64)  PRIMARY KEY,
      query_text CLOB          NOT NULL,
      candidates CLOB,
      chosen     CLOB,
      outcome    VARCHAR2(32),
      created_ts TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
    )
  ';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE != -955 THEN RAISE; END IF;
END;
/
