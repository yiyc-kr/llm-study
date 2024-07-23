import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Token, Parenthesis
from sqlparse.tokens import Keyword, DDL, Whitespace, Comment

# Placeholder function for LLM-based comment generation
def llm_generate(text):
    # Remove leading parts up to the last '_'
    processed_text = text.split('_')[-1]
    # Replace specific uppercase starting letters with specific words
    if processed_text.startswith('T'):
        processed_text = 'TABLE ' + processed_text[1:]
    elif processed_text.startswith('C'):
        processed_text = 'COLUMN ' + processed_text[1:]
    return f"Generated comment for {processed_text}"

def process_create_statement(statement, comments):
    create_stmt = ""
    tokens = list(statement.flatten())
    table_name = ""
    schema_name = ""
    column_comments = {}

    for token in tokens:
        if token.ttype is Keyword.DDL and token.value.upper() == 'CREATE':
            create_stmt += f"{token.value} TABLE "
            table_name_token = next(tokens)
            if isinstance(table_name_token, Identifier):
                table_name = table_name_token.get_real_name()
                if '.' in table_name:
                    schema_name, table_name = table_name.split('.')
                create_stmt += f"{schema_name + '.' if schema_name else ''}{table_name} "
                full_table_name = f"{schema_name + '.' if schema_name else ''}{table_name}"
                if full_table_name in comments:
                    create_stmt += f"-- {comments[full_table_name]}\n"
        elif token.ttype not in Whitespace and token.ttype not in Comment:
            create_stmt += f"{token.value} "
        
        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                col_name = identifier.get_real_name()
                col_type = identifier.token_next(0).value
                column_comments[col_name] = col_type

    # Add comments for columns
    for col_name, col_type in column_comments.items():
        full_column_name = f"{schema_name + '.' if schema_name else ''}{table_name}.{col_name}"
        if full_column_name in comments:
            create_stmt = create_stmt.replace(f"{col_name} {col_type}", f"{col_name} {col_type} -- {comments[full_column_name]}")

    return create_stmt.strip()

def process_comment_statement(statement):
    comment_stmt = {}
    tokens = iter(statement.tokens)
    next(tokens)  # Skip 'COMMENT'
    next(tokens)  # Skip 'ON'
    target_type = next(tokens).value.upper()
    target_name = next(tokens).get_real_name()
    comment = next(tokens).get_real_name().strip("'")
    
    if target_type == 'TABLE':
        comment_stmt[target_name] = llm_generate(comment)
    elif target_type == 'COLUMN':
        comment_stmt[target_name] = llm_generate(comment)

    return comment_stmt

def parse_ddl(ddl):
    statements = sqlparse.parse(ddl)
    create_statements = []
    comments = {}

    for statement in statements:
        if statement.get_type() == 'CREATE':
            create_statements.append(statement)
        elif statement.get_type() == 'UNKNOWN' and statement.token_first().match(Keyword, 'COMMENT'):
            comments.update(process_comment_statement(statement))

    transformed_ddl = ""
    for create_statement in create_statements:
        transformed_ddl += process_create_statement(create_statement, comments) + "\n\n"

    return transformed_ddl.strip()

# Example DDL statement with comments and additional options
ddl = """
CREATE TABLE example (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    age INT
) TABLESPACE mytablespace PCTFREE 10 INITRANS 1 STORAGE (INITIAL 10M NEXT 50M) NOLOGGING NOPARALLEL;

COMMENT ON TABLE example IS 'This_is_an_example_table.';
COMMENT ON COLUMN example.id IS 'Identifier_for_the_example_table.';
COMMENT ON COLUMN example.name IS 'Name_of_the_person.';
COMMENT ON COLUMN example.age IS 'Age_of_the_person.';

CREATE TABLE schema.example (
    id INT PRIMARY KEY,
    description TEXT
) TABLESPACE mytablespace PCTFREE 10 INITRANS 1 STORAGE (INITIAL 10M NEXT 50M) NOLOGGING NOPARALLEL;

COMMENT ON TABLE schema.example IS 'This_is_an_example_table_with_schema.';
COMMENT ON COLUMN schema.example.id IS 'Identifier_for_the_schema_example_table.';
COMMENT ON COLUMN schema.example.description IS 'Description_of_the_example.';
"""

transformed_ddl = parse_ddl(ddl)
print(transformed_ddl)
