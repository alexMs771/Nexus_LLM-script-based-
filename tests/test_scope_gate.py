from app.services.scope_gate import classify_request


def test_scope_gate_distinguishes_python_boundary_cases():
    assert classify_request("What is Python?").intent == "OUT_OF_SCOPE"
    assert classify_request("Create a table to store Python package names and versions.").intent == "IN_SCOPE"


def test_scope_gate_requires_details_for_generic_table_request():
    assert classify_request("Create a table.").intent == "NEEDS_CLARIFICATION"


def test_scope_gate_blocks_destructive_request():
    assert classify_request("Drop all production tables.").intent == "RESTRICTED_OR_DESTRUCTIVE"


def test_scope_gate_distinguishes_unsupported_sql_from_destructive_ddl():
    assert classify_request("Select employees with salary greater than 1000.").intent == "UNSUPPORTED_SQL_OPERATION"
    assert classify_request("Delete inactive customers with a SQL statement.").intent == "UNSUPPORTED_SQL_OPERATION"
