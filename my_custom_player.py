from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation
    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.
    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def __init__(self, player_id):
        super().__init__(player_id)

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least
        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.
        See RandomPlayer and GreedyPlayer in sample_players for more examples.
        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
        my_depth = 1
        depth_limit = 16
        while my_depth<=depth_limit:
            if state.ply_count < 2:
                self.queue.put(random.choice(state.actions()))
                break
            if state.ply_count < 4:
                self.queue.put(self.alpha_beta_search(state, my_depth))
                break
            else:
                if len(state.actions()) == 1:
                    self.queue.put(state.actions()[0])
                else:
                    move = self.alpha_beta_search(state, my_depth)
                    if move:
                        self.queue.put(move)
                    else:
                        self.queue.put(random.choice(state.actions()))
                        break
            my_depth += 1

    def alpha_beta_search(self, gameState, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.
        You can ignore the special case of calling this function
        from a terminal state.
        """
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for action in gameState.actions():
            value = self.min_value(gameState.result(action), alpha, beta, depth - 1)
            alpha = max(alpha, value)
            if value > best_score:
                best_score = value
                best_move = action
        return best_move

    def min_value(self, gameState, alpha, beta, depth):
        if gameState.terminal_test():
            return gameState.utility(self.player_id)

        if depth <= 0:
            return self.score(gameState)

        value = float("inf")
        for action in gameState.actions():
            value = min(value, self.max_value(gameState.result(action), alpha, beta, depth - 1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def max_value(self, gameState, alpha, beta, depth):
        if gameState.terminal_test():
            return gameState.utility(self.player_id)

        if depth <= 0:
            return self.score(gameState)

        value = float("-inf")
        for action in gameState.actions():
            value = max(value, self.min_value(gameState.result(action), alpha, beta, depth - 1))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def score(self, state):
        heuristic_choice = "progressively offensive"
        ply_count = state.ply_count
        player_location = state.locs[self.player_id]
        opponent_player_id = 0
        if self.player_id == 0:
            opponent_player_id = 1
        opponent_location = state.locs[opponent_player_id]
        player_liberties = len(state.liberties(player_location))
        opponent_liberties = len(state.liberties(opponent_location))
        game_change_point = 10

        if heuristic_choice == "baseline":
            return player_liberties - opponent_liberties #this is for the benchmark tests
        elif heuristic_choice == "weighted offensive":
            return player_liberties - 2 * opponent_liberties #first custom heuristic favors less opponent freedom
        elif heuristic_choice == "progressively defensive":
            return player_liberties * ply_count - opponent_liberties # this heuristic increases the importance of number of moves left for the player 
        elif heuristic_choice == "progressively offensive":
            return player_liberties - ply_count * opponent_liberties # this heuristic increases the importance of number of moves left for the player 
        elif heuristic_choice == "defensive to offensive": # in this heuristics strategy changes from defensive to offensive after 10 moves
            if ply_count < game_change_point:
                return player_liberties * 2 - opponent_liberties
            else:
                return player_liberties - 2 * opponent_liberties
        elif heuristic_choice == "weighted liberty": # this heuristic favors locations close to center by valuing them 3 times more than others.
            temp = 0
            for location in state.liberties(player_location):
                if location < 66 and location >33:
                    temp += 3
                else:
                    temp += 1
            return temp
                    