-- scripts/sql/006_skillbox_governance.sql
-- Add governance columns to SKILLBOX. Idempotent: ORA-01430 (column already
-- exists) is swallowed so the script is safe to re-run. Existing rows are
-- backfilled to 'active' so prior behaviour (every listed skill usable) holds.
DECLARE
  e_col_exists EXCEPTION;
  PRAGMA EXCEPTION_INIT(e_col_exists, -1430);
  PROCEDURE add_col(ddl VARCHAR2) IS
  BEGIN
    EXECUTE IMMEDIATE ddl;
  EXCEPTION WHEN e_col_exists THEN NULL;
  END;
BEGIN
  add_col('ALTER TABLE SKILLBOX ADD (status VARCHAR2(16) DEFAULT ''pending'')');
  add_col('ALTER TABLE SKILLBOX ADD (source VARCHAR2(16))');
  add_col('ALTER TABLE SKILLBOX ADD (created_day NUMBER)');
  -- promoted: a standard skill is marked promoted once an approved enhanced
  -- skill supersedes it for the same topic, so it is excluded from retrieval and
  -- the agent never sees two recipes for the same job. NUMBER(1): 0 false, 1 true.
  add_col('ALTER TABLE SKILLBOX ADD (promoted NUMBER(1) DEFAULT 0)');
  EXECUTE IMMEDIATE 'UPDATE SKILLBOX SET status = ''active'' WHERE status IS NULL';
  EXECUTE IMMEDIATE 'UPDATE SKILLBOX SET promoted = 0 WHERE promoted IS NULL';
  COMMIT;
END;
/
