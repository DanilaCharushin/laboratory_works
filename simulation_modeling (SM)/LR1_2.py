from random import randint


def generate_arrival_time():
    return randint(1, 11)


def generate_processing_time():
    return randint(1, 19)


def generate_computer_id():
    return randint(1, 2)


def set_task_for_computer(queue, current_time):
    task = queue.pop(0)
    task["start_processing_time"] = current_time
    return task


def generate_task(arrival_time, task_id):
    return {
        "arrival_time": arrival_time,
        "start_processing_time": -1,
        "processing_time": generate_processing_time(),
        "task_id": task_id,
    }


def calc_in_system_time(task):
    return (
        task["processing_time"] + 1 +
        task["start_processing_time"] -
        task["arrival_time"]
    )


def main():
    task_number = 1000
    experiment_number = 100
    
    total_processing_time_1 = 0
    total_processing_time_2 = 0
    total_wait_time = 0
    total_downtime_1 = 0
    total_downtime_2 = 0
    total_time = 0
    total_in_system_time = 0
    
    for _ in range(experiment_number):
        processing_task_1 = {}
        processing_task_2 = {}

        queue = []

        busy_1 = False
        busy_2 = False

        task_count = 0
        current_time = 0

        arrival_time = generate_arrival_time()

        while True:
            if task_count < task_number and current_time == arrival_time:
                task_count += 1
                queue.append(generate_task(arrival_time, task_count))
                arrival_time = generate_arrival_time() + current_time
            # ======================================================================================================
            if not busy_1 and not busy_2 and len(queue) != 0:
                computer_id = generate_computer_id()
                if computer_id == 1:
                    processing_task_1 = set_task_for_computer(queue, current_time)
                    busy_1 = True
                    total_in_system_time += calc_in_system_time(processing_task_1)
                else:
                    processing_task_2 = set_task_for_computer(queue, current_time)
                    busy_2 = True
                    total_in_system_time += calc_in_system_time(processing_task_2)
            # ======================================================================================================
            if not busy_1 and busy_2 and len(queue) != 0:
                processing_task_1 = set_task_for_computer(queue, current_time)
                busy_1 = True
                total_in_system_time += calc_in_system_time(processing_task_1)
            # ======================================================================================================
            if not busy_2 and busy_1 and len(queue) != 0:
                processing_task_2 = set_task_for_computer(queue, current_time)
                busy_2 = True
                total_in_system_time += calc_in_system_time(processing_task_2)
            # ======================================================================================================
            if busy_1:
                if processing_task_1["start_processing_time"] + processing_task_1["processing_time"] == current_time:
                    busy_1 = False
                else:
                    total_processing_time_1 += 1
            # ======================================================================================================
            if busy_2:
                if processing_task_2["start_processing_time"] + processing_task_2["processing_time"] == current_time:
                    busy_2 = False
                else:
                    total_processing_time_2 += 1
            # ======================================================================================================
            if not busy_1:
                total_downtime_1 += 1
            if not busy_2:
                total_downtime_2 += 1
            if len(queue) != 0:
                total_wait_time += 1
            # ======================================================================================================
            current_time += 1

            if task_count == task_number and not busy_1 and not busy_2 and len(queue) == 0:
                break

        total_time += current_time

    print("===============RESULTS===============")
    print(f'TOTAL TIME: {total_time}')
    print(f'TOTAL PROCESSING TIME 1: {total_processing_time_1}')
    print(f'TOTAL PROCESSING TIME 2: {total_processing_time_2}')
    print(f'TOTAL WAIT TIME: {total_wait_time}')
    print(f'TOTAL DOWNTIME 1: {total_downtime_1}')
    print(f'TOTAL DOWNTIME 2: {total_downtime_2}')
    print("=====================================")
    print(f'AVERAGE WAIT TIME: {round(total_wait_time / (task_number * experiment_number), 1)}')
    print(f'AVERAGE BEING IN SYSTEM TIME: {round(total_in_system_time / (task_number * experiment_number), 1)}')
    print(f'DOWNTIME 1 PROBABILITY: {round(100 * total_downtime_1 / total_time, 1)}%')
    print(f'DOWNTIME 2 PROBABILITY: {round(100 * total_downtime_2 / total_time, 1)}%')


if __name__ == "__main__":
    main()
