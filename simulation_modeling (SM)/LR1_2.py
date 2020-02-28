from random import randint

# ЗАДАНИЕ:
# Обслуживание компьютерами равновероятно, очередь заданий не ограничена
# Интервал поступления заданий: (1, 11)
# Интервал обработки заданий: (1, 19)

# СТАТИСТИКА:
# среднее время нахождения задания в очереди;
# среднее время пребывания задания в системе;
# вероятности простоя компьютеров

# количество задач в эксперимент
TASK_NUMBER = 1000
# количество экспериментов
EXPERIMENT_NUMBER = 100


# генератор времени поступления
def generate_arrival_time():
    return randint(1, 11)


# генератор времени обработки
def generate_processing_time():
    return randint(1, 19)


# генератор номера компьютера
def generate_computer_id():
    return randint(1, 2)


TOTAL_PROCESSING_TIME_1 = 0
TOTAL_PROCESSING_TIME_2 = 0

TOTAL_WAIT_TIME = 0

TOTAL_DOWNTIME_1 = 0
TOTAL_DOWNTIME_2 = 0

TOTAL_TIME = 0
TOTAL_IN_SYSTEM_TIME = 0


def set_task_for_computer(queue, current_time):
    task = queue.pop(0)
    task["start_processing_time"] = current_time
    return task


def generate_task(arrival_time, task_id):
    task = {
        "arrival_time": arrival_time,
        "start_processing_time": -1,
        "processing_time": generate_processing_time(),
        "task_id": task_id,
    }
    return task


def increment_total_in_system_time(task):
    global TOTAL_IN_SYSTEM_TIME
    TOTAL_IN_SYSTEM_TIME += (
        task["processing_time"] + 1 +
        task["start_processing_time"] -
        task["arrival_time"]
    )


for _ in range(EXPERIMENT_NUMBER):
    processing_task_1 = {}
    processing_task_2 = {}

    queue = []

    busy_1 = False
    busy_2 = False

    task_count = 0
    current_time = 0

    arrival_time = generate_arrival_time()

    while True:
        if task_count < TASK_NUMBER and current_time == arrival_time:
            task_count += 1
            queue.append(generate_task(arrival_time, task_count))
            arrival_time = generate_arrival_time() + current_time
        # ======================================================================================================
        if not busy_1 and not busy_2 and len(queue) != 0:
            computer_id = generate_computer_id()
            if computer_id == 1:
                processing_task_1 = set_task_for_computer(queue, current_time)
                busy_1 = True
                increment_total_in_system_time(processing_task_1)
            else:
                processing_task_2 = set_task_for_computer(queue, current_time)
                busy_2 = True
                increment_total_in_system_time(processing_task_2)
        # ======================================================================================================
        if not busy_1 and busy_2 and len(queue) != 0:
            processing_task_1 = set_task_for_computer(queue, current_time)
            busy_1 = True
            increment_total_in_system_time(processing_task_1)
        # ======================================================================================================
        if not busy_2 and busy_1 and len(queue) != 0:
            processing_task_2 = set_task_for_computer(queue, current_time)
            busy_2 = True
            increment_total_in_system_time(processing_task_2)
        # ======================================================================================================
        if busy_1:
            if processing_task_1["start_processing_time"] + processing_task_1["processing_time"] == current_time:
                busy_1 = False
            else:
                TOTAL_PROCESSING_TIME_1 += 1
        # ======================================================================================================
        if busy_2:
            if processing_task_2["start_processing_time"] + processing_task_2["processing_time"] == current_time:
                busy_2 = False
            else:
                TOTAL_PROCESSING_TIME_2 += 1
        # ======================================================================================================
        if not busy_1:
            TOTAL_DOWNTIME_1 += 1
        if not busy_2:
            TOTAL_DOWNTIME_2 += 1
        if len(queue) != 0:
            TOTAL_WAIT_TIME += 1
        # ======================================================================================================
        current_time += 1

        if task_count == TASK_NUMBER and not busy_1 and not busy_2 and len(queue) == 0:
            break

    TOTAL_TIME += current_time

print("===============RESULTS===============")
print(f'TOTAL TIME: {TOTAL_TIME}')
print(f'TOTAL PROCESSING TIME 1: {TOTAL_PROCESSING_TIME_1}')
print(f'TOTAL PROCESSING TIME 2: {TOTAL_PROCESSING_TIME_2}')
print(f'TOTAL WAIT TIME: {TOTAL_WAIT_TIME}')
print(f'TOTAL DOWNTIME 1: {TOTAL_DOWNTIME_1}')
print(f'TOTAL DOWNTIME 2: {TOTAL_DOWNTIME_2}')
print("=====================================")
print(f'AVERAGE WAIT TIME: {TOTAL_WAIT_TIME / (TASK_NUMBER * EXPERIMENT_NUMBER)}')
print(f'AVERAGE BEING IN SYSTEM TIME: {TOTAL_IN_SYSTEM_TIME / (TASK_NUMBER * EXPERIMENT_NUMBER)}')
print(f'DOWNTIME 1 PROBABILITY: {round(100 * TOTAL_DOWNTIME_1 / TOTAL_TIME, 2)}%')
print(f'DOWNTIME 2 PROBABILITY: {round(100 * TOTAL_DOWNTIME_2 / TOTAL_TIME, 2)}%')