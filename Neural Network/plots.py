import matplotlib.pyplot as plt


epoch = range(1,13)

val_acc = [0.8766, 0.8836, 0.8901, 0.8939, 0.8953, 0.8954, 0.8971, 0.8985, 0.8977, 0.8998, 0.9024, 0.8998]

loss = [0.4024, 0.3231, 0.3064, 0.2952, 0.2884, 0.2818, 0.2783, 0.2744, 0.2708, 0.2694, 0.2671, 0.2643]

acc = [0.8403, 0.8710, 0.8787, 0.8843, 0.8879, 0.8901, 0.8917, 0.8931, 0.8948, 0.8958, 0.8969, 0.8976]

val_loss = [0.3028, 0.2926, 0.2832, 0.2699, 0.2664, 0.2645, 0.2582, 0.2555, 0.2590, 0.2511, 0.2478, 0.2494]



def percent(list):
	return map(lambda x: 100*x, list)

val_acc = percent(val_acc)
loss = percent(loss)
acc = percent(acc)
val_loss = percent(val_loss)

line1, = plt.plot(epoch, val_acc, label='val_acc')
line2, = plt.plot(epoch, acc, label= 'acc')
plt.legend(handles=[line1, line2])

plt.axis([1, 12, 83, 92])
plt.xticks(range(1,13))
plt.title("acc and val_acc per epoch")
plt.xlabel("Number of epoch")
plt.ylabel("% loss")


plt.show()
