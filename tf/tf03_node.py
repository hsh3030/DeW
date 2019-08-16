import tensorflow as tf

node1 = tf.constant(3.0, tf.float32) 
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1 : ", node1, "node2 : ", node2)
print("node3 : ", node3)
# 노드형식으로 분류되어 session으로 사용자가 볼 수 있게 출력 할 수 있다.
# graph를 구성하고 나면 Session 오브젝트를 만들어서 graph를 실행할 수 있습니다. op 생성함수에서 다른 graph를 지정해줄 때까지는 default graph가 Session에서 실행됩니다
sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3) : ", sess.run(node3))