-- scripts/sql/007_enhanced_skillbox.sql
-- The enhanced skill store: engine-inducted, versioned, governed skills,
-- SEPARATE from the base SKILLBOX (the agent queries both). Idempotent:
-- ORA-00955 (name already used) is swallowed so re-runs are safe.
DECLARE
  e_tbl_exists EXCEPTION;
  PRAGMA EXCEPTION_INIT(e_tbl_exists, -955);
BEGIN
  EXECUTE IMMEDIATE '
    CREATE TABLE ENHANCED_SKILLBOX (
      skill_id         VARCHAR2(64) PRIMARY KEY,
      topic            VARCHAR2(128) NOT NULL,
      name             VARCHAR2(256) NOT NULL,
      description      CLOB,
      when_to_use      CLOB,
      steps            CLOB,
      skills_used      CLOB,
      likely_tools     CLOB,
      errors_and_fixes CLOB,
      provenance       CLOB,
      embedding        VECTOR,
      status           VARCHAR2(16) DEFAULT ''pending'',
      source           VARCHAR2(16) DEFAULT ''unified'',
      version          NUMBER DEFAULT 1,
      created_day      NUMBER,
      review_comment   CLOB,
      created_ts       TIMESTAMP WITH TIME ZONE
    )';
EXCEPTION WHEN e_tbl_exists THEN NULL;
END;
/

-- Existing course databases need the governance comment column too.
DECLARE
  e_col_exists EXCEPTION;
  PRAGMA EXCEPTION_INIT(e_col_exists, -1430);
BEGIN
  EXECUTE IMMEDIATE
    'ALTER TABLE ENHANCED_SKILLBOX ADD (review_comment CLOB)';
EXCEPTION WHEN e_col_exists THEN NULL;
END;
/
