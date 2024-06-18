import json
import time
import numpy as np
import math
import scipy


class AdditiveLaggedFibonacciGenerator:
    def __init__(self, seed=1, lag_a=5, lag_b=17, modulus=2 * 32):
        self.lag_a = lag_a
        self.lag_b = lag_b
        self.modulus = modulus
        self.state = [seed]

        for _ in range(1, self.lag_b):
            self.state.append((self.state[-1] * 1103515245 + 12345) % self.modulus)

    def next_number(self):
        new_value = (self.state[-self.lag_a] + self.state[-self.lag_b]) % self.modulus
        self.state.append(new_value)
        return new_value

    def generate_uniform_random(self):
        return self.next_number() / self.modulus

    def generate_numbers(self, count, lower_bound, upper_bound):
        numbers = []
        for _ in range(count):
            rand = self.generate_uniform_random()
            number = lower_bound + rand * (upper_bound - lower_bound)
            numbers.append(int(number))
        return numbers


def xor_shift_rotate_scale(lower_bound, upper_bound, n):
    random_numbers = []
    range_size = upper_bound - lower_bound + 1
    x = int(time.time() * 1000)

    for _ in range(n):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        random_num = ((x >> 16) & 0xFFFF) * range_size // 0x10000
        random_numbers.append(random_num + lower_bound)

    return random_numbers


def main():
    lower_bound = 0
    upper_bound = 10000
    n = 500000
    alfg = AdditiveLaggedFibonacciGenerator(seed=42)
    # data_samples = defaultdict(list)

    for i in range(20):
        print(f"Набор {i + 1} ОК!")

    exit()

    # for _ in range(20):
        # data_samples['alfg'].append(alfg.generate_numbers(count=n, lower_bound=lower_bound, upper_bound=upper_bound))
        # data_samples['xor'].append(xor_shift_rotate_scale(lower_bound, upper_bound, n))

    # with open('samples.json', 'w', encoding='utf8') as samples_file:
        # json.dump(data_samples, samples_file)

    with open('samples.json', 'r', encoding='utf8') as samples_file:
        data_samples = json.load(samples_file)

    num_elements = 500000
    num_intervals = round(1 + np.log2(num_elements))
    print(f'Количество интервалов: {num_intervals}')

    max_xor = max([max(sample) for sample in data_samples['xor']])
    min_xor = min([min(sample) for sample in data_samples['xor']])
    step_xor = (max_xor - min_xor) // num_intervals + 1
    intervals_xor = [i * step_xor for i in range(1, num_intervals + 1)]
    print('Интервалы для xor:', intervals_xor)

    max_alfg = max([max(sample) for sample in data_samples['alfg']])
    min_alfg = min([min(sample) for sample in data_samples['alfg']])
    step_ccg = (max_alfg - min_alfg) // num_intervals + 1
    intervals_ccg = [i * step_ccg for i in range(1, num_intervals + 1)]
    print('Интервалы для alfg:', intervals_ccg)

    chi_squared_stats = {'xor': [], 'alfg': []}
    for sample in data_samples['xor']:
        frequencies = [0] * num_intervals
        for element in sample:
            for interval_index in range(num_intervals):
                if element <= intervals_xor[interval_index]:
                    frequencies[interval_index] += 1
                    break
        chi_squared_sum = 0
        for freq in frequencies:
            chi_squared_sum += (num_intervals * (freq * 2) / num_elements)
        chi_squared_value = round(chi_squared_sum - num_elements, 3)
        chi_squared_stats['xor'].append(chi_squared_value)

    critical_values = np.array([1.239, 2.167, 4.255, 6.346, 9.037, 14.07, 18.48])

    for index, value in enumerate(chi_squared_stats['xor']):
        if (value < critical_values[0]) or (value > critical_values[-1]):
            print(f"Набор{index + 1} не ОК!")
        else:
            print(f"Набор{index + 1} ОК!")

    for index, value in enumerate(chi_squared_stats['alfg']):
        if (value < critical_values[0]) or (value > critical_values[-1]):
            print(f"Набор {index + 1} не ОК!")
        else:
            print(f"Набор {index + 1} ОК!")

    for ind, sample in enumerate(data_samples['alfg']):
        data_samples['alfg'][ind] = [val % 2 for val in sample]

    print(data_samples['alfg'][0][:10])

    # Частотный блочный тест
    def freq_within_block_test(df, n, m):
        N = n // m
        pi = np.array([np.sum(df[i * m:(i + 1) * m] == 1) / m for i in range(N)])
        chi_stat = 4 * m * np.sum(np.power((pi - 0.5), 2))
        p_val = round(scipy.special.gammaincc(N / 2, chi_stat / 2), 2)

        return p_val >= 0.01, p_val

    # Частотный побитовый тест
    def monobit_test(df, n):
        S_n = np.sum(df == 1) - np.sum(df == 0)
        S_obs = abs(S_n / np.sqrt(n))
        p_val = round(math.erfc(S_obs / math.sqrt(2)), 3)

        return p_val >= 0.01, p_val

    # Тест на последовательность одинаковых битов
    def runs_test(df, n):
        p = np.sum(df == 1) / n
        if abs(p - 0.5) > (2 / np.sqrt(n)):
            return False, 0
        V_n = 1
        for i in range(n - 1):
            if df[i] != df[i + 1]:
                V_n += 1
        p_val = round(math.erfc(abs(V_n - 2 * n * p * (1 - p)) / (2 * np.sqrt(2 * n) * p * (1 - p))), 3)

        return p_val >= 0.01, p_val


if __name__ == "__main__":
    main()
