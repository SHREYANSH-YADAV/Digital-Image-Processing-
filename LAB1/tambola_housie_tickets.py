"""
Tambola (Housie) Ticket Generator
----------------------------------
Generates valid Tambola/Housie tickets following standard rules:
  - Each ticket has 3 rows and 9 columns.
  - Each row contains exactly 5 numbers and 4 blanks.
  - Column 1: numbers 1-9
    Column 2: numbers 10-19
    Column 3: numbers 20-29
    ...
    Column 9: numbers 80-90
  - Every column has at least 1 number.
  - Numbers within a column are sorted top to bottom.
"""

import random


def generate_ticket():
    col_counts = [1] * 9
    remaining = 15 - 9
    while remaining > 0:
        col = random.randint(0, 8)
        if col_counts[col] < 3: 
            col_counts[col] += 1
            remaining -= 1

    col_slots = []
    for col, count in enumerate(col_counts):
        col_slots.extend([col] * count)

    row_counts = [0, 0, 0]
    assignment = {}  
    for col in range(9):
        assignment[col] = []

    random.shuffle(col_slots)
    for col in col_slots:
        candidates = [r for r in range(3) if row_counts[r] < 5 and r not in assignment[col]]
        if not candidates:
            return generate_ticket()
        row = random.choice(candidates)
        assignment[col].append(row)
        row_counts[row] += 1

    grid = [[0] * 9 for _ in range(3)]
    for col in range(9):
        low = 1 if col == 0 else col * 10
        high = 9 if col == 0 else (col * 10 + 9 if col < 8 else 90)
        count = len(assignment[col])
        numbers = sorted(random.sample(range(low, high + 1), count))
        rows_for_col = sorted(assignment[col])
        for r, num in zip(rows_for_col, numbers):
            grid[r][col] = num

    return grid


def print_ticket(grid):
    print("+" + "----+" * 9)
    for row in grid:
        line = "|"
        for val in row:
            line += f"{val:>3} |" if val != 0 else "    |"
        print(line)
    print("+" + "----+" * 9)


def generate_multiple_tickets(n=1):
    return [generate_ticket() for _ in range(n)]


if __name__ == "__main__":
    num_tickets = int(input("Enter the number of tickets to generate: "))  # change this to generate more/fewer tickets
    tickets = generate_multiple_tickets(num_tickets)

    for i, ticket in enumerate(tickets, start=1):
        print(f"\nTicket #{i}")
        print_ticket(ticket)