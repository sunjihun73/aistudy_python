
import tensorflow as tf

def mlFunc(name):
    w = tf.Variable(tf.random.normal([1]), name = "w")
    b = tf.Variable(tf.random.normal([1]), name = "b")

    X = tf.constant([1, 2, 3], dtype=tf.float32)
    Y = tf.constant([1, 2, 3], dtype=tf.float32)

    def hypothesis():
        return w * X + b

    def cost():
        return tf.reduce_mean(tf.square(hypothesis() - Y))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    print("학습횟수     weight     bias      cost")

    for step in range(2001):
        with tf.GradientTape() as tape:
            current_cost = cost()

        grads = tape.gradient(current_cost, [w, b])
        optimizer.apply_gradients(zip(grads, [w, b]))

        if step % 100 == 0:
            print(
                format(step, "4d"),
                " ",
                format(w.numpy()[0], "10.4f"),
                " ",
                format(b.numpy()[0], "8.4f"),
                format(current_cost.numpy(), "10.5f")
            )

    print("\n사과를 5개 사면?", format(hypothesis().numpy()[0] * 5, "5.3f"), "원")


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    mlFunc('sunjihun')