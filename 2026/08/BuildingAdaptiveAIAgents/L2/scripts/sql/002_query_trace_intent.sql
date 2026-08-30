-- 002_query_trace_intent.sql
-- Add an optional INTENT column to the course-owned QUERY_TRACES table.
--
-- Module 2's skill-induction harvester groups successful traces by intent and
-- synthesizes one procedural skill per intent. The natural-language label
-- ("find_vessels_by_destination", "check_weather_window", ...) is captured by
-- module_2_token_space/scripts/run.py at trace-generation time and consumed by
-- course_lab.skill_induction.induce_skills.
--
-- QUERY_TRACES is course-owned (created by scripts/sql/002_query_traces.sql),
-- not by the oracleagentmemory package, so an in-place ALTER is the right move
-- — no side table or join required.
--
-- Idempotency: ORA-01430 (column being added already exists) is swallowed so
-- this script is safe to re-run.

BEGIN
  EXECUTE IMMEDIATE 'ALTER TABLE QUERY_TRACES ADD (intent VARCHAR2(128))';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE != -1430 THEN RAISE; END IF;
END;
/
