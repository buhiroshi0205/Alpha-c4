import numpy as np
import math

class node:

	def __init__(self, state, value, parent):
		self.parent = parent
		self.state = state
		self.visits = 1
		self.value = value
		self.children = []

	def get_parent(self):
		return self.parent

	def add_children(self, children):
		self.children.append(children)

	def get_children(self):
		return self.children

	def has_children(self):
		return len(self.children) != 0

	def select_next_node(self):
		length = len(self.children)
		UCB = np.empty(length) # Upper Confidence Bound
		if self.state.turn == 1:
			for i in range(length):
				UCB[i] = self.children[i].value + math.sqrt(math.log(self.visits)/self.children[i].visits/6)
			return self.children[np.argmax(UCB)]
		else:
			for i in range(length):
				UCB[i] = self.children[i].value - math.sqrt(math.log(self.visits)/self.children[i].visits/6)
			return self.children[np.argmin(UCB)]

	def update(self, visits):
		self.visits += visits
		if not self.has_children(): return
		length = len(self.children)
		values = np.empty(length)
		for i in range(length):
			values[i] = self.children[i].value
		if self.state.turn == 1:
			self.value = max(values)
		else:
			self.value = min(values)

	def select_move(self):
		length = len(self.children)
		LCB = np.empty(length) # Lower Confidence Bound
		if self.state.turn == 1:
			for i in range(length):
				LCB[i] = self.value - math.sqrt(math.log(self.visits)/self.children[i].visits/6)
			return self.children[np.argmax(LCB)].state.last_move
		else:
			for i in range(length):
				LCB[i] = self.value + math.sqrt(math.log(self.visits)/self.children[i].visits/6)
			return self.children[np.argmin(LCB)].state.last_move