# https://stackoverflow.com/a/62074206
import io, tokenize, re


def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += " " * (start_col - last_col)
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = "\n".join(l for l in out.splitlines() if l.strip())
    return out


if __name__ == "__main__":
    code = '''
def add(a, b):
    """
    Add two numbers together.

    Args:
        a (int or float): First number.
        b (int or float): Second number.

    Returns:
        int or float: The sum of a and b.
    """
    # Perform addition
    result = a + b
    
    # Return the result
    return result


def subtract(a, b):
    """
    Subtract the second number from the first.

    Args:
        a (int or float): First number.
        b (int or float): Second number.

    Returns:
        int or float: The result of a - b.
    """
    # Perform subtraction
    result = a - b
    
    # Return the result
    return result


if __name__ == "__main__":
    # Test the add function
    sum_result = add(10, 5)
    
    # Test the subtract function
    diff_result = subtract(10, 5)
    
    # Print the results
    print("Sum:", sum_result)  # Should print Sum: 15
    print("Difference:", diff_result)  # Should print Difference: 5
'''

    print(remove_comments_and_docstrings(code))
    # open("temp.py", "w").write(remove_comments_and_docstrings(code))
