# Модель: Метод Зейделя (5 семестр)
# Автор: Унтілова Євгенія, група АІ-233

from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

def gauss_seidel(A, b, x0, eps=1e-3, max_iter=100):

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)

    n = len(b)

    for _ in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])

            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new.tolist()

        x = x_new

    return x.tolist()

@app.route('/calculate', methods=['GET'])
def calculate():

    # Значення x з URL
    x = float(request.args.get('x', 0))

    # Система рівнянь
    A = [
        [16, 2, 8],
        [2, 13, -4],
        [-10, 6, 26]
    ]

    b = [-8, -2, -9]

    # Початкове наближення
    x0 = [x, x, x]

    result = gauss_seidel(A, b, x0)

    return jsonify({
        "input": x,
        "method": "Gauss-Seidel",
        "result": result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)