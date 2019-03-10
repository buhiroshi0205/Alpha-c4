import numpy as np
import time, graphviz
import node as nd
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Flatten, Dropout


REPLAY_MEMORY_SIZE = 300
BATCH = 32


class ai:

	def __init__(self, think_time=0.1, weights=None):

		self.think_time = think_time
		self.model = Sequential()
		self.model.add(Conv3D(64, (1,4,4), padding='same', input_shape=(1,2,7,6), data_format='channels_first', activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Conv3D(64, (1,4,4), padding='same', activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Flatten())
		self.model.add(Dense(256, activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(256, activation='relu'))
		self.model.add(Dense(1, activation='tanh'))
		self.model.compile('rmsprop', 'mse')
		if weights != None:
			self.model.load_weights(weights)
		self.model.predict_on_batch(np.array([np.zeros((1,2,7,6))]))

		self.input_memory = np.empty((1000, 1,2,7,6))
		self.label_memory = np.empty((1000, 1))
		self.memory_index = 0
		self.full_data = False

		self.inputs = []
		self.values = []
		self.turns = []

		self.model.summary()

	def get_move(self, state):
		root = nd.node(state, 0, None)
		self.UCT(root)

		# store data
		self.inputs.append(state.get_processed_board())
		self.values.append(root.value)
		self.turns.append(root.state.turn)

		return root.select_move()

	def get_eval(self, state):
		root = nd.node(state, 0, None)
		self.UCT(root)
		return root.value

	def UCT(self, root):
		count = 0
		maxdepth = 1
		end_time = time.time() + self.think_time
		while time.time() < end_time:
			node = root

			# select
			depth = 1
			while node.has_children():
				node = node.select_next_node()
				depth += 1
			if depth > maxdepth: maxdepth = depth

			# expand and evaluate
			if node.state.terminal:
				branches = 1
			else:
				legal_moves = node.state.get_legal_moves()
				branches = len(legal_moves)
				count += branches
				batch = []
				states = []
				for i in legal_moves:
					new_state = node.state.move(i)
					states.append(new_state)
					batch.append(new_state.get_processed_board())
				values = self.model.predict_on_batch(np.array(batch)) * states[0].turn
				for i in range(branches):
					if states[i].check_game():
						values[i] = states[i].get_winner()
					node.add_children(nd.node(states[i], values[i], node))
					
			# backpropagate
			node.update(branches)
			parent = node.get_parent()
			while parent != None:
				parent.update(len(states))
				parent = parent.get_parent()
		print('node count = %d, maxdepth = %d' % (count, maxdepth))


	def flush_data(self, winner):
		length = len(self.values)
		print(self.values)
		for i in range(length):
			self.input_memory[(i+self.memory_index) % REPLAY_MEMORY_SIZE] = self.inputs[i]
			self.label_memory[(i+self.memory_index) % REPLAY_MEMORY_SIZE] = (self.values[i] + winner) / 2 * self.turns[i]
		if self.memory_index + length >= REPLAY_MEMORY_SIZE:
			self.full_data = True
			self.memory_index += length - REPLAY_MEMORY_SIZE
		else:
			self.memory_index += length
		self.inputs, self.values, self.turns = [], [], []

	def train(self):
		if not self.full_data: return
		indices = np.random.randint(0, REPLAY_MEMORY_SIZE, BATCH)
		self.model.train_on_batch(self.input_memory[indices],  self.label_memory[indices])

	def save_weights(self, name='weights.h5'):
		self.model.save_weights(name)

	def display_tree(self, root):
		graph = graphviz.Digraph()
		self.id = 0
		label = str(root.state.get_raw_board()) + '\n' + str(root.value) + '\n' + str(root.visits)
		graph.node(str(id), label)
		self.id += 1
		self.add_child_nodes(graph, root, 0)
		graph.view(cleanup=True)

	def add_child_nodes(self, graph, node, thisid):
		if not node.has_children(): return
		for child in node.get_children():
			label = str(child.state.get_raw_board()) + '\n' + str(child.value) + '\n' + str(child.visits)
			childid = self.id
			graph.node(str(self.id), label)
			self.id += 1
			self.add_child_nodes(graph, child, childid)
			graph.edge(str(thisid), str(childid))
