import tensorflow as tf
a=tf.constant(2,name="a")
b=tf.constant(3,name="b")
c=tf.add(a,b,name="c")
with tf.Session() as sess:
    writer=tf.summary.FileWriter("./myoutput",sess.graph)
    print(sess.run(c))
    
writer.close()