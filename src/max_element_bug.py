import random

random_numbers = [random.randint(1, 10000) for _ in range(20)]

print(f"Random numbers are:", random_numbers)

def comp(pair):
    return pair[0] > pair[1]

def max_element_falsch(arr, comp):
    best_index = 0
    for i in range(1, len(arr)):
        if comp((arr[i], arr[best_index])):
            best_index = i
            print(f"Best_index atm is:", best_index)
    return best_index

def max_element_richtig(arr, comp):
    best_index = 0
    for i in range(len(arr) - 1):
        if comp((arr[best_index], arr[best_index + 1])):
            best_index = i
            print(f"Best_index atm is:", best_index)
    return best_index

print(f"Falscher Ansatz:")
max_element_falsch(random_numbers, comp)
print(f"Richtiger Ansatz:")
max_element_richtig(random_numbers, comp)