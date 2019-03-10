import ai as uct
import state

THINK_TIME = 0.5
START_EPISODE = 1600
EPISODES = 400
SAVE_INTERVAL = 100

ai = uct.ai(THINK_TIME, 'weights_{}.h5'.format(START_EPISODE)) if START_EPISODE else uct.ai(THINK_TIME)

for episode in range(START_EPISODE, START_EPISODE + EPISODES):
	game_over = False
	game = state.state()
	moves = []
	while game_over == False:
		#print game.get_raw_board()
		move = ai.get_move(game)
		moves.append(move)
		game = game.move(move)
		game_over = game.check_game()
	print(moves)
	ai.flush_data(game.get_winner())
	ai.train()
	#raw_input('press enter to continue...')
	if (episode+1) % SAVE_INTERVAL == 0:
		ai.save_weights('weights_%d.h5' % (episode+1))

