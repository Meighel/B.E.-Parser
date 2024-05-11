import re

class Position:
    def __init__(self, idx, ln, col):
        self.idx = idx
        self.ln = ln
        self.col = col

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == "\n":
            self.ln += 1
            self.col = 0
        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col)

class Token:
    def __init__(self, type, value=None, start=None, end=None, position=None):
        self.type = type 
        self.value = value
        self.position = position
        self.start = start
        self.end = end

    def __repr__(self):
        if self.value:
            return f"{self.type}:{self.value} ({self.start}-{self.end})"
        return f"({self.type})"

class BooleanLexer:
    def __init__(self, user_expression):
        self.user_expression = user_expression

    @staticmethod
    def make_tokens(user_expression):
        tokens = []
        patterns = [
            (r'AND|OR', 'OPERATOR'),
            (r'NOT', 'NEGATE'),
            (r'A|B|C', 'VARIABLE'),
            (r'1|0', 'BINARY'),
            (r'\(', 'LPAREN'),
            (r'\)', 'RPAREN')
        ]
        combined_patterns = '|'.join(f'(?P<{token_type}>{pattern})' for pattern, token_type in patterns)

        for match in re.finditer(combined_patterns, user_expression):
            for token_type, token_value in match.groupdict().items():
                if token_value is not None:
                    start = match.start(token_type)
                    end = match.end(token_type)
                    position = Position(match.start(token_type), 1, match.start(token_type) + 1)
                    tokens.append(Token(token_type, token_value, start, end, position))

        return tokens

class VBNode:
    def __init__(self, tok):
        self.tok = tok

    def __repr__(self):
        return f"{self.tok.value}"

class OperatorNode:
    def __init__(self, left_node, op_tok, right_node=None):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

    def __repr__(self):
        return f"({str(self.left_node)} {self.op_tok.value} {str(self.right_node)})"

class UnaryNode:
    def __init__(self, not_tok, node):
        self.not_tok = not_tok
        self.node = node

    def __repr__(self):
        return f'{self.not_tok.value}{self.node}'

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self):
        res = self.expr()
        return res

    def factor(self):
        tok = self.current_tok

        if tok.type == 'NEGATE':
            self.advance()  # Consume the 'NOT'
            node = self.factor()  # Parse the variable after 'NOT'
            return UnaryNode(tok, node)
        if tok.type == 'LPAREN':
            self.advance()  # Consume the '('
            node = self.expr()  # Parse the expression inside the parentheses
            if self.current_tok.type != 'RPAREN':
                raise Exception("Expected ')'")
            self.advance()  # Consume the ')'
            return node
        elif tok.type == 'VARIABLE' or tok.type == 'BINARY':
            self.advance()
            return VBNode(tok)
        else:
            raise Exception(f"Unexpected token: {tok}")

    def term(self):
        left = self.factor()

        while self.current_tok.type == 'OPERATOR' and self.current_tok.value == 'AND':
            op_tok = self.current_tok
            self.advance()
            right = self.factor()
            left = OperatorNode(left, op_tok, right)

        return left

    def expr(self):
        left = self.term()

        while self.current_tok and self.current_tok.type == 'OPERATOR' and self.current_tok.value == 'OR':
            op_tok = self.current_tok
            self.advance()
            right = self.term()
            left = OperatorNode(left, op_tok, right)

        return left

class Interpreter:
    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node)

    def no_visit_method(self, node):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_VBNode(self, node):
        print("Found Variable/Binary node:", node.tok.value)

    def visit_UnaryNode(self, node):
        print("Found Unary Node: NOT")
        self.visit(node.node)

    def visit_OperatorNode(self, node):
        print(f"Found Operator node: {node.op_tok.value}")
        self.visit(node.left_node)
        if node.right_node:
            self.visit(node.right_node)

    @staticmethod
    def remove_continuous_not(node):
        if isinstance(node, UnaryNode) and node.not_tok.value == 'NOT':
            inner_node = node.node
            if isinstance(inner_node, UnaryNode) and inner_node.not_tok.value == 'NOT':
                inner_node = inner_node.node

            if inner_node is not node.node:
                return inner_node
            else:
                return UnaryNode(node.not_tok, inner_node)

        if isinstance(node, OperatorNode):
            node.left_node = Interpreter.remove_continuous_not(node.left_node)
            if node.right_node:
                node.right_node = Interpreter.remove_continuous_not(node.right_node)

        return node


    def rules_of_b_algebra(self, node):
        if isinstance(node, OperatorNode):
            if node.op_tok.value == "OR":
                if isinstance(node.right_node, VBNode) and node.right_node.tok.value == '0':
                    return node.left_node  # Apply identity law: A OR 0 = A
                elif isinstance(node.right_node, VBNode) and node.right_node.tok.value == '1':
                    return node.right_node  # Apply annulment law: A OR 1 = 1
                elif isinstance(node.left_node, VBNode) and isinstance(node.right_node, VBNode) and node.left_node.tok.value == node.right_node.tok.value:
                    return node.left_node  # Apply idempotent law: A OR A = A
                elif isinstance(node.left_node, VBNode) and isinstance(node.right_node, UnaryNode) and node.left_node.tok.value == node.right_node.node.tok.value:
                    return VBNode(Token('BINARY', '1'))  # Apply complement law: A OR (NOT A) = 1
                elif isinstance(node.left_node, VBNode) and isinstance(node.right_node, OperatorNode) and node.right_node.op_tok.value == "AND" and isinstance(node.right_node.left_node, UnaryNode) and node.right_node.left_node.not_tok.value == "NOT" and node.left_node.tok.value == node.right_node.left_node.node.tok.value:
                    return OperatorNode(node.left_node, node.op_tok, node.right_node.right_node)  # Apply rule no. 11: A OR ((NOT A) AND B) = A OR B
                elif isinstance(node.left_node, VBNode) and isinstance(node.right_node, OperatorNode) and node.right_node.op_tok.value == "AND" and node.left_node.tok.value == node.right_node.left_node.tok.value:
                    return node.left_node  # Apply absorptive law: A OR (A AND B) = A

            elif node.op_tok.value == "AND":
                if isinstance(node.right_node, VBNode) and node.right_node.tok.value == '0':
                    return node.right_node  # Apply annulment law: A AND 0 = 0
                elif isinstance(node.right_node, VBNode) and node.right_node.tok.value == '1':
                    return node.left_node  # Apply identity law: A AND 1 = A
                elif isinstance(node.left_node, VBNode) and isinstance(node.right_node, VBNode) and node.left_node.tok.value == node.right_node.tok.value:
                    return node.left_node  # Apply idempotent law: A AND A = A
                elif isinstance(node.left_node, VBNode) and isinstance(node.right_node, UnaryNode) and node.left_node.tok.value == node.right_node.node.tok.value:
                    return VBNode(Token('BINARY', '0'))  # Apply complement law: A AND (NOT A) = 0
                elif isinstance(node.left_node, OperatorNode) and node.right_node.op_tok.value == "OR" and node.left_node.op_tok.value == "OR" and isinstance(node.left_node.left_node, VBNode) and isinstance(node.left_node.right_node, VBNode) and isinstance(node.right_node.right_node, VBNode) and node.right_node.left_node.tok.value == node.left_node.left_node.tok.value:
                    return OperatorNode(node.left_node.left_node, node.right_node.op_tok, OperatorNode(node.left_node.right_node, node.op_tok, node.right_node.right_node))  # Apply rule no. 12: (A OR B) AND (A OR C) = A OR (B AND C)
    
        return node

    def simplify_expression(self, node):
        # Keep track of the previous node for comparison
        prev_node = None

        while prev_node != node:
            prev_node = node

            # Recursively simplify the left and right nodes
            if isinstance(node, OperatorNode):
                node.left_node = self.simplify_expression(node.left_node)
                if node.right_node:
                    node.right_node = self.simplify_expression(node.right_node)

                # Apply rules of boolean algebra to simplify the expression
                node = self.remove_continuous_not(node)
                node = self.rules_of_b_algebra(node)

        return node



def run(user_expression):
    tokens = BooleanLexer.make_tokens(user_expression)
    for token in tokens:
        print(f"The index of {token.value} is {token.start}")

    parser = Parser(tokens)
    ast = parser.parse()

    interpreter = Interpreter()

    simplified_ast = interpreter.simplify_expression(ast)

    return tokens, ast, simplified_ast

if __name__ == '__main__':
    while True:
        user_expression = input("Enter an Expression: ")
        tokens, ast, simplified_ast = run(user_expression)

        # Print tokens
        print("\nTokens:")
        for token in tokens:
            print(token)

        # Print AST
        print("\nAST:")
        print(ast)

        # Print Simplified AST
        print("\nSimplified AST:")
        print(simplified_ast)
