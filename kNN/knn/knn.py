from queue import PriorityQueue

def predict(X_data, Y_data, example, distance, k):
    prediction = 0
    pq = PriorityQueue()
    for i in range(len(X_data)):
        row = X_data[i, :]
        d = distance(row, example)
        if pq.size() < k:
            pq.push(d, Y_data[i])
        elif d < pq.peek_priority():
            pq.pop()
            pq.push(d, Y_data[i])

    prediction = pq.pop()
    item = pq.pop()
    while item != None:
        prediction = (prediction + item) / 2.0
        item = pq.pop()

    return round(prediction)
