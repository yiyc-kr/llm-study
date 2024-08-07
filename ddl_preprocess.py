# pip install transformers==4.43.4

import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Token, Parenthesis, TokenList
from sqlparse.tokens import Keyword, DDL, Whitespace, Comment, Newline, Name, String
from collections import defaultdict
import re

import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("nllb_finetuned_ko2en", token=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("nllb_finetuned_ko2en", token=True, src_lang="kor_Hang")


# check statement if it is create table statement
def is_create_table_statement(statement):
    for token in statement.tokens:
        if token.ttype == sqlparse.tokens.Keyword and token.value.upper() == "TABLE":
            return True
        elif isinstance(token, TokenList):
            if is_create_table_statement(token):
                return True
    return False

# get simple create table statement
def get_simple_create_table_statement(statement):
    i = 0
    start_i = 0
    stop_newline_condition = 1
    for token in statement.tokens:
        if stop_newline_condition and token.ttype == Newline:
            start_i += 1
        else:
            stop_newline_condition = 0
        i += 1
        if isinstance(token, Parenthesis):
            break
    statement.tokens = statement.tokens[start_i:i]
    return statement

# filter ddl statements to get create table statements
def filter_create_table_statements(statements):
    list_create_table_statements = []
    for statement in statements:
        if statement.get_type().upper() == "CREATE" and is_create_table_statement(statement):
            statement = get_simple_create_table_statement(statement)
            list_create_table_statements.append(statement)

    return list_create_table_statements

# preprocess comment statement
def preprocess_comment_statement(statement, is_col=True):
    table_name = None
    col_name = None
    comment = None
    for token in statement.tokens:
        if token.ttype == String.Single:
            comment = str(token).strip("'")
        elif isinstance(token, Identifier):
            if is_col:
                table_name = '.'.join(str(token).split('.')[:-1])
                col_name = '.'.join(str(token).split('.')[-1:])
            else:
                table_name = str(token)
    return table_name, comment, col_name

# translate comment(ko -> en)
def translate_comment(comment):
    # Remove leading parts up to the last '_'
    comment = comment.split("_")[-1]
    # Replace specific uppercase starting letters with specific words
    if comment.startswith("IDD"):
        comment = comment[3:]
    elif comment.startswith("ABC"):
        comment = "에이비씨" + comment[3:]
    
    inputs = tokenizer(comment, return_tensors="pt").to("cuda")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.encode("eng_Latn")[0], max_length=512)
    translate_comment = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    return translate_comment

# get table comment statements and column comment statements
def get_table_col_comment_statements(statements):
    dict_table_comment_statements = defaultdict(str)
    dict_col_comment_statements = defaultdict(dict)
    start_i = 0
    stop_newline_condition = 1

    for statement in statements:
        if statement.get_type().upper() == "UNKNOWN" and statement.token_first().match(Keyword, "COMMENT"):
            for token in statement.tokens:
                if stop_newline_condition and token.ttype == Newline:
                    start_i += 1
                else:
                    stop_newline_condition = 0
                if token.ttype == Keyword and token.value.upper() == "TABLE":
                    statement.tokens = statement.tokens[start_i:]
                    table_name, table_comment, _ = preprocess_comment_statement(statement, is_col=False)
                    dict_table_comment_statements[table_name] = translate_comment(table_comment)
                    break
                elif token.ttype == Keyword and token.value.upper() == "COLUMN":
                    statement.tokens = statement.tokens[start_i:]
                    table_name, col_comment, col_name = preprocess_comment_statement(statement, is_col=True)
                    dict_col_comment_statements[table_name][col_name] = translate_comment(col_comment)
                    break

    return dict_table_comment_statements, dict_col_comment_statements

# get ddl preprocessed(translated, parsed)
def get_preprocessed_ddl(ddl):
    ddl = ddl.replace("\"", "")
    statements = sqlparse.parse(ddl)

    list_create_table_statements = filter_create_table_statements(statements)
    dict_table_comment_statements, dict_col_comment_statements = get_table_col_comment_statements(statements)

    table_name_pattern = r"CREATE TABLE\s+([\w.]+)"
    for statement in list_create_table_statements:
        list_statement_line = str(statement).split("\n")
        match = re.search(table_name_pattern, list_statement_line[0])
        table_name = match.group(1)
        list_statement_line[0] += f" -- {dict_table_comment_statements[table_name]}"
        print(list_statement_line[0])
        for index, statement_line in enumerate(list_statement_line[1:], start=1):
            col_name = statement_line.split()[0]
            if dict_col_comment_statements[table_name][col_name] == {}:
                print(list_statement_line[index])
                continue
            list_statement_line[index] += f" -- {dict_col_comment_statements[table_name][col_name]}"
            print(list_statement_line[index])
