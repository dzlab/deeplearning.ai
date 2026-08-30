-- 004_voyage_manifest.sql
-- Course-owned VOYAGE_MANIFEST table: the structured, queryable manifest of the
-- KNOWN voyages the Module 3 / Module 4 demos ask the models about.
-- Run as dmuser against FREEPDB1.
--
-- Why this table exists:
--   The voyages the models reference (VY-1000.. known; VY-7xxx/VY-8xxx unknown)
--   were previously generated in-memory by make_voyages() and never persisted,
--   so a learner could not verify against the DB whether a referenced voyage
--   actually exists. This table makes existence checkable: a voyage IS in the
--   manifest  => the model should ANSWER it; a voyage is ABSENT => the model
--   should REFUSE it ("No record of voyage VY-XXXX in the manifest").
--
--   Only KNOWN voyages (make_voyages(), VY-1000..) are seeded here. The held-out
--   eval ids (VY-8xxx) and train-time unknowns (VY-7xxx) are deliberately absent
--   — their absence is the ground truth the refusal demo is checked against.
--
-- The oracleagentmemory package creates its own DLAI_* tables via
-- SchemaPolicy.CREATE_IF_NECESSARY — do NOT create those here. This file manages
-- only the course-extra manifest table.
--
-- Idempotency: ORA-00955 (name already used by an existing object) is silently
-- swallowed so the script is safe to re-run.

BEGIN
  EXECUTE IMMEDIATE '
    CREATE TABLE VOYAGE_MANIFEST (
      vid       VARCHAR2(16)  PRIMARY KEY,
      vessel    VARCHAR2(128) NOT NULL,
      cargo     VARCHAR2(128) NOT NULL,
      origin    VARCHAR2(64)  NOT NULL,
      dest      VARCHAR2(64)  NOT NULL,
      seeded_ts TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
    )
  ';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE != -955 THEN RAISE; END IF;
END;
/
