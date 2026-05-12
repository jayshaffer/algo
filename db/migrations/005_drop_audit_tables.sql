-- 005_drop_audit_tables.sql
--
-- Drops the audit-related tables. The audit is being rewritten as a
-- Claude Code /loop driven by docs/audit-playbook.md; Jira is the
-- canonical store for findings now.
--
-- audit_llm_calls FK-references audit_runs, so drop order matters
-- (or use CASCADE).
--
-- Spec: docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md

DROP TABLE IF EXISTS audit_llm_calls CASCADE;
DROP TABLE IF EXISTS audit_findings CASCADE;
DROP TABLE IF EXISTS audit_runs CASCADE;
