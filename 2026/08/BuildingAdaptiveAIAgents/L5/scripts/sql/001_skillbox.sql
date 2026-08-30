-- 001_skillbox.sql
-- Course-owned SKILLBOX table with a VECTOR(384, FLOAT32) embedding column.
-- Run as dmuser against FREEPDB1.
--
-- The oracleagentmemory package creates its own DLAI_* tables via
-- SchemaPolicy.CREATE_IF_NECESSARY — do NOT create those here.
-- This file manages only the course-extra tables.
--
-- Idempotency: ORA-00955 (name already used by an existing object) is silently
-- swallowed so the script is safe to re-run.

BEGIN
  EXECUTE IMMEDIATE '
    CREATE TABLE SKILLBOX (
      skill_id   VARCHAR2(64)               PRIMARY KEY,
      name       VARCHAR2(256)              NOT NULL,
      description CLOB,
      recipe_steps CLOB,
      provenance  CLOB,
      embedding  VECTOR(384, FLOAT32),
      created_ts TIMESTAMP                 DEFAULT CURRENT_TIMESTAMP
    )
  ';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE != -955 THEN RAISE; END IF;
END;
/

-- HNSW vector index for cosine similarity search.
-- Wrapped in a PL/SQL block so that on Oracle Free images that do not support
-- HNSW the error is logged and execution continues rather than aborting the
-- entire bootstrap.
BEGIN
  EXECUTE IMMEDIATE '
    CREATE VECTOR INDEX skillbox_embedding_idx
      ON SKILLBOX(embedding)
      ORGANIZATION INMEMORY NEIGHBOR GRAPH
      DISTANCE COSINE
      WITH TARGET ACCURACY 95
  ';
EXCEPTION
  WHEN OTHERS THEN
    -- ORA-00955: index already exists — silent.
    -- Any other error (e.g. feature not supported in this image): log and continue.
    IF SQLCODE != -955 THEN
      DBMS_OUTPUT.PUT_LINE('WARNING: HNSW index creation skipped: ' || SQLERRM);
    END IF;
END;
/
