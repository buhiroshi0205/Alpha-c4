import numpy as np

class state:

	def __init__(self, board=np.zeros((7,6), int), heights=np.zeros(7, int), turn=1, last_move=None):
		self.board = board
		self.heights = heights
		self.turn = turn
		self.last_move = last_move
		self.terminal = False

	def move(self, move):
		board, heights = np.copy(self.board), np.copy(self.heights)
		board[move][heights[move]] = self.turn
		heights[move] += 1
		return state(board, heights, self.turn * -1, move)

	def get_legal_moves(self):
		return [i for i in range(7) if self.heights[i] != 6]

	def get_raw_board(self):
		return self.board

	def get_processed_board(self):
		return np.array([np.stack((self.board == self.turn, self.board == -self.turn))])

	def check_game(self):
		column = self.last_move
		row = self.heights[self.last_move] - 1
		turn = self.turn * -1
		game_over = False
		# check vertical win
		if (not game_over) and row > 2:
			for i in range(row, row-4, -1):
				if self.board[column][i] != turn: break
			else:
				game_over = True
				#print 'vertical win detected'
		#check horizontal win
		if not game_over:
			consecutive = 1
			for i in range(column-1, -1, -1):
				if self.board[i][row] == turn:
					consecutive += 1
				else:
					break
			for i in range(column+1, 7):
				if self.board[i][row] == turn:
					consecutive += 1
				else:
					break
			if consecutive >= 4:
				game_over = True
				#print 'horizontal win detected'
		# check diagonal win - SW to NE
		if (not game_over) and (-3 < column-row < 4):
			consecutive = 1
			for i in range(1, min(7-column, 6-row)):
				if self.board[column+i][row+i] == turn:
					consecutive += 1
				else:
					break
			for i in range(1, min(column, row)+1):
				if self.board[column-i][row-i] == turn:
					consecutive += 1
				else:
					break
			if consecutive >= 4:
				game_over = True
				#print 'diagonal win 1 detected'
		# check diagonal win - NW to  SE
		if (not game_over) and (2 < column+row < 9):
			consecutive = 1
			for i in range(1, min(7-column, row+1)):
				if self.board[column+i][row-i] == turn:
					consecutive += 1
				else:
					break
			for i in range(1, min(column+1, 6-row)):
				if self.board[column-i][row+i] == turn:
					consecutive += 1
				else:
					break
			if consecutive >= 4:
				game_over = True
				#print 'diagonal win 2 detected'

		# determine game outcome
		if game_over: 
			self.winner = turn
		else:
			# check tie
			for i in self.heights:
				if i != 6: break
			else:
				game_over = True
				self.winner = 0
		self.terminal = game_over
		return game_over

	def get_winner(self):
		return self.winner