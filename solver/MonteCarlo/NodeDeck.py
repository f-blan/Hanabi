#from markupsafe import re
from torch import rand
from solver.Fuzzy.FDeck import FDeck
#from solver.MonteCarlo.MCPlayer import MCPlayer
from numpy.random import randint
import numpy as np



#from solver.MonteCarlo.MCMove import MCMove

class NodeDeck(FDeck):
    def __init__(self, main_player, n_players: int):
        super().__init__(None, n_players)
        self.turns_until_end = -1
        self.endgame = False

        #data structure to keep track of random decisions on deck (ex. when removing a random card we track it here)
        self.randomMask = np.zeros((5,5), dtype=np.int16)
        #data structure to keep track of which cards can be in our main play
        self.hintMask = np.zeros((5,5)) == 0
        #self.deck = np.array(self.deck, dtype=np.float32)
        #self.cards_in_game= np.array(self.cards_in_game, dtype=np.float32)

    def RemoveCards(self, to_remove: np.ndarray):
        #print("----------__REMOVAL------------------")
        #print(self.deck)
        super().RemoveCards(to_remove)
        if np.any(self.deck<0):
            print("------PRE EXCEPTION----")
            print(f"deck:\n{self.deck}\ncards in game:\n{self.cards_in_game}\nto_remove:\n{to_remove}\nrandom mask:\n{self.randomMask}")

            #ad hoc fix
            assert self.randomMask[to_remove[0],to_remove[1]] >= 1
            targets = np.logical_and(self.deck>=1, self.randomMask<=-1)
            assert np.any(targets)
            xr,yr = targets.nonzero()

            index = randint(0, xr.shape[0])
            x = xr[index]
            y = yr[index]
            self.randomMask[x,y] +=2

            self.cards_in_game[x,y] -= 1
            self.deck[x,y] -= 1
            self.cards_in_game[to_remove[0], to_remove[1]]+=1
            self.deck[to_remove[0], to_remove[1]]+=1
            
        
        assert np.any(self.deck<0)==False
        #print(self.deck)
        #print("------END REMOVAL------")
    
    def RemoveCardsFromGame(self, to_remove: np.ndarray):
        return super().RemoveCardsFromGame(to_remove)
    
    def RemoveHardUnknownFromGame(self):
        #it's meaningless updating the matrix. Statistics changes caused by using an hard unknown should have no impact
        self.n_cards_in_game-=1
        assert self.n_cards_in_game > 1
    
    def RemoveHardUnkown(self):
        #same as RemoveHardUnknownFromGame
        self.n_cards_in_deck -= 1
        assert self.n_cards_in_game >= 0
    
    def RemoveHintedCard(self,  hint: np.ndarray):
        cont = True
        #we may know something about the card we need to remove
        m = hint>=2

        if np.any(m) and cont:
            #perfect knowledge
            x,y = m.nonzero()
            x=x[0]
            y=y[0]
            if self.deck[x,y] < 1:
                #we previously reoved this card for randomness, remove another one
                #some old debug prints
                print("---PRE EXCEPTION---")
                print(f"deck:\n{self.deck}\nhint:\n{hint}\nmask:\n{self.mask}\nrandomMask:\n{self.randomMask}")
                    
                #assert self.randomMask[x,y] >= 1
                targets = np.logical_and(self.deck>=1, self.randomMask<=-1)
                assert np.any(targets)
                xr,yr = targets.nonzero()

                index = randint(0, xr.shape[0])
                x = xr[index]
                y = yr[index]
                self.randomMask[x,y] +=2

            self.cards_in_game[x,y] -= 1
            self.deck[x,y] -= 1
            cont = False

        #partial/no knowledge: remove one of the cards this card can be (randomly)
        m = hint > 0
        if np.any(m) and cont:
            #partial knowledge
            x,y = m.nonzero()

            mask_filter = self.mask[x, y]
            x = x[mask_filter]
            y = y[mask_filter]
            
            self.randomMask[x,y]-=1
            if x.shape[0] == 0:
            
                print("---PRE EXCEPTION---")
                print(f"deck:\n{self.deck}\nhint:\n{hint}\nmask:\n{self.mask}\nrandomMask:\n {self.randomMask}")
                #ad hoc fix: partial knowledge narrows the possibiities only to cards that have been removed to randomness. Replace this
                #past random removal with another one

                self.replace_randomicity(m)
            else:
                index = randint(0, x.shape[0])
                x = x[index]
                y = y[index]
                self.randomMask[x,y]+=2
        
                self.deck[x,y] -= 1
                self.cards_in_game[x,y] -= 1

            cont = False
        
        m = hint < 0
        if np.any(m) and cont:
            #we only have negative knowledge
            possible_cards = np.logical_not(m)
            x,y = possible_cards.nonzero()

            mask_filter = self.mask[x, y]
            x = x[mask_filter]
            y = y[mask_filter]
            
            self.randomMask[x,y]-=1

            if x.shape[0] == 0:
                print("---PRE EXCEPTION---")
                print(f"deck:\n{self.deck}\nhint:\n{hint}\nmask:\n{self.mask}\nrandomMask:\n{self.randomMask}")
                self.replace_randomicity(m)
            else:
                index = randint(0, x.shape[0])
                x = x[index]
                y = y[index]
            
                self.randomMask[x,y]+=2

                self.deck[x,y] -= 1
                self.cards_in_game[x,y] -= 1
            cont = False
        
        if cont:
            #we know nothing of the card
            x,y = self.mask.nonzero()
            self.randomMask[x,y]-=1
            if x.shape[0] == 0:
                print("---PRE EXCEPTION---")
                print(f"deck:\n{self.deck}\nhint:\n{hint}\nmask:\n{self.mask}\nrandomMask:\n{self.randomMask}")
                self.replace_randomicity(self.mask)
            else:
                index = randint(0, x.shape[0])
                x = x[index]
                y = y[index]
            
                self.randomMask[x,y]+=2

                self.deck[x,y] -= 1
                self.cards_in_game[x,y] -= 1
            cont = False
        

        self.mask = self.deck > 0
        #self.deck = self.clean_matrix(self.deck)
        #self.cards_in_game = self.clean_matrix(self.cards_in_game)

        if np.any(self.deck<0):
            print("------PRE EXCEPTION----")
            print(f"deck:\n{self.deck}\ncards in game:\n{self.cards_in_game}\nhint:\n{hint}")
        
        assert np.any(self.deck<0)==False

        self.n_cards_in_deck-=1
        self.n_cards_in_game-=1

    def update_endgame_condition(self, main_player):
        if self.n_cards_in_deck - main_player.n_cards <= 0 and self.endgame == False:
            self.endgame = True
            self.turns_until_end = self.n_players
        elif self.endgame == True:
            self.turns_until_end -=1

    def update_expected_values(self, fireworks: np.ndarray,  main_player):
        
        self.update_death_line(fireworks)
        
        #compute the filtered deck (estimate of cards in deck only)
        #if (lastMove.type <= 1 and lastMove.playerOrder == main_player.order) or (lastMove.type == 2 and lastMove.destination == main_player.order and lastMove.known):
            #do it only if we gathered some new informations or the main player's hand has changed
        self.update_filtered_deck(main_player)
        self.filtered_deck = self.clean_matrix(self.filtered_deck)

        

        if self.n_cards_in_filtered_deck <= 0:
            return 
        #compute playability and discardability of hard unknown cards 
        needed_values = fireworks +1
        needed_colors = np.array([i for i in range(0,5)])

        needed_colors= needed_colors[needed_values<5]
        needed_values= needed_values[needed_values<5]

        #playability = number of playable cards in deck/ total number of cards in deck
        n_playable = np.sum(self.filtered_deck[needed_values, needed_colors])
        self.playability_rc = n_playable/self.n_cards_in_filtered_deck
        
        #discardability = sum of discardabilities of cards in deck /total number of cards in deck
        #discardabilities = np.zeros((25,25), dtype= np.float32)

        tot_discardability = 0
        arange = np.arange(5)
        #for each color
        for i in range(0,needed_colors.shape[0]):
            discardable_indexes = arange<needed_values[i]
            dead_indexes = arange >= self.death_line[i]
            non_trivial_indexes = np.logical_not(np.logical_or(discardable_indexes,dead_indexes))
            #trivial: cards that are lower than firework of their color have discardability 1
            tot_discardability += np.sum(self.filtered_deck[discardable_indexes, i])

            #trivial: cards that are beyond the death line
            tot_discardability += np.sum(self.filtered_deck[dead_indexes, i])

            #for the rest, compute their discardability through formula at the end of evaluate_known_cards()
            non_trivials = self.filtered_deck[non_trivial_indexes, i]
            
            cards_in_game = self.cards_in_game[non_trivial_indexes, i]
            non_finished_indexes = cards_in_game > 0
            non_trivials=non_trivials[non_finished_indexes]
            cards_in_game=cards_in_game[non_finished_indexes]

            #discardability of each card * number of cards of that kind
            nt_discardabilities = self.discardability_fn(cards_in_game)*non_trivials
                    #1- np.reciprocal(self.cards_in_game[arange>=needed_values[i], i]))*non_trivials
            tot_discardability += np.sum(nt_discardabilities)
        
        self.discardability_rc = tot_discardability/self.n_cards_in_filtered_deck


    
    def evaluate_known_cards(self, to_evaluate: np.ndarray, fireworks: np.ndarray, hard_unknowns: np.ndarray, n_cards:int):
        #we either know exactly these cards or they are hard unknowns
        regular_indexes = hard_unknowns == False
        arange = np.array([i for i in range(0, len(hard_unknowns))]) < n_cards
        regular_indexes = np.logical_and(arange,regular_indexes)
        
        ret_p = np.zeros((to_evaluate.shape[1]),dtype=np.float32)
        ret_d = np.zeros((to_evaluate.shape[1]), dtype=np.float32)
        regulars = to_evaluate[:,regular_indexes]
        #print(f"to_evaluate: {to_evaluate}\nregular_indexes: {regular_indexes},\nregulars: {regulars},\nret_p: {ret_p},\nret_d:{ret_d}")

        ret_p[regular_indexes], ret_d[regular_indexes] =super().evaluate_known_cards(regulars, fireworks)


        #for hard unknowns just return the expected values
        ret_p[hard_unknowns] = self.playability_rc
        ret_d[hard_unknowns] = self.discardability_rc

        return ret_p, ret_d

    def evaluate_unknown_cards(self, n_cards: int, fireworks: np.ndarray, hints: np.ndarray, hard_unknowns):
        #called for cards that should be evaluated only through hints
        #hard_unknowns=hard_unknowns[0:n_cards]
        #hints = hints[0:n_cards]
        regular_indexes = hard_unknowns == False
        arange = np.array([i for i in range(0, len(hard_unknowns))]) < n_cards
        regular_indexes = np.logical_and(arange,regular_indexes)
        ret_p = np.zeros((len(regular_indexes)), dtype=np.float32)
        ret_d = np.zeros((len(regular_indexes)), dtype=np.float32)
        

        ret_p[regular_indexes], ret_d[regular_indexes] = super().evaluate_unknown_cards(ret_p[regular_indexes].shape[0], fireworks, 
                                                                                        hints[:,:, regular_indexes])

        #for hard unknowns just return the expected values
        ret_p[hard_unknowns] = self.playability_rc
        ret_d[hard_unknowns] = self.discardability_rc

        return ret_p, ret_d

    def clean_matrix(self, matrix: np.ndarray):
        """
            Take a matrix representing cards, take all values < 0 and distribute them among positive ones
        """
        neg = matrix < 0
        if np.any(neg):
            #print("--------CLEANING UP---------")
            neg_x, neg_y = neg.nonzero()
            pos_x, pos_y = (matrix>0).nonzero()

            #print(f"negative: {neg_x}, {neg_y} positive: {pos_x}, {pos_y}")
            #print(filtered_deck)

            to_add = -np.sum(matrix[neg_x,neg_y])
            if pos_x.shape[0] > 0:
                to_add = to_add/pos_x.shape[0]
            else:
                to_add = 0
            
            #print(f"to_add: {to_add}")
            matrix[pos_x,pos_y]+= to_add
            matrix[neg_x,neg_y] = 0
        return matrix 

    
    def update_filtered_deck(self, main_player):
        #get filtered deck (an estimate of the cards in the deck computed with agent's hints)
        filtered_deck = np.array(self.deck, dtype=np.float32)
        self.n_cards_in_filtered_deck = self.n_cards_in_deck
        for i in range(0, main_player.cardsInHand):
            if self.n_cards_in_filtered_deck <= 0:
                break

            if main_player.hard_unknowns[i] == True:
                continue

            hint = main_player.hints[:,:, i]
            
            #process perfect knowledge
            kn = hint >= 2
            if np.any(kn):
                #we know what this card is. Remove it from filtered deck
                x, y = kn.nonzero()
                filtered_deck[x[0], y[0]] -= 1
                self.n_cards_in_filtered_deck -=1
                continue
            #process partial knowledge
            kn = hint > 0
            if np.any(kn):
                #we know something. Remove a total of 1 card distributed among the possibilities
                x, y = kn.nonzero()

                #filter out cards that can't be in the deck or in the agent's hand because we saw them in the game
                #example effect: we know this card is a 3, but we know that there is only a 3 of a given color in 
                #deck + agent's hand. to_remove will be a full 1 and we will completely remove all 3 from filtered_deck
                mask_filter = self.mask[x, y]
                x = x[mask_filter]
                y = y[mask_filter]

                if x.shape[0] == 0:
                    #this happens when partial knowledge narrows the possibility to one or more card, but those cards was
                    #removed through randomicity: remove one of the cards that wasn't removed due to randomicity
                    #instead.
                    #some old debug prints
                    print("---PRE EXCEPTION---")
                    print(f"deck:\n{self.deck}\nhint:\n{hint}\nmask:\n{self.mask}\nplayer h_tracker:\n{main_player.hint_tracker}\ni:{i}\nrandomMask:\n{self.randomMask}")
                    targets = np.logical_and(self.deck>=1, self.randomMask <= -1)
                    assert np.any(targets)
                    x,y = targets.nonzero()

                    #adjust deck, cards in game and mask
                    self.deck[x,y] -=1
                    self.cards_in_game[x,y]-=1
                    xr, yr = kn.nonzero()
                    xr = xr[0]
                    yr = yr[0]

                    self.deck[xr,yr] += 1
                    self.cards_in_game[xr,yr] += 1
                    filtered_deck[xr,yr] = 0
                    self.n_cards_in_filtered_deck-=1
                    
                    continue 

                #you are removing one card in total: distribute it 
                to_remove = 1/x.shape[0]
                filtered_deck[x,y] -= to_remove
                self.n_cards_in_filtered_deck -=1
                continue
            #process negative knowledge
            kn = hint < 0
            if np.any(kn):
                #we know what this card is not: remove a total of 1 card distributed among all the cards this card can be
                possible_cards = np.logical_not(kn)
                x,y = possible_cards.nonzero()
                
                mask_filter = self.mask[x, y]
                #print(f"deck:{self.deck}\nmask: {self.mask}\nmask_filter:\n {mask_filter}\nx and y: {x}-{y}")
                x = x[mask_filter]
                y = y[mask_filter]

                if x.shape[0] == 0:
                    #some old debug prints
                    print("---PRE EXCEPTION---")
                    print(f"deck:\n{self.deck}\nhint:\n{hint}\nmask:\n{self.mask}\nplayer h_tracker:\n{main_player.hint_tracker}\ni:{i}\nrandomMask:\n{self.randomMask}")
                    targets = np.logical_and(self.deck>=1, self.randomMask <= -1)
                    assert np.any(targets)
                    x,y = targets.nonzero()

                    #adjust deck, cards in game and mask
                    self.deck[x,y] -=1
                    self.cards_in_game[x,y]-=1
                    xr, yr = kn.nonzero()
                    xr = xr[0]
                    yr = yr[0]

                    self.deck[xr,yr] += 1
                    self.cards_in_game[xr,yr] += 1
                    filtered_deck[xr,yr] = 0
                    self.n_cards_in_filtered_deck-=1
                    continue

                to_remove = 1/x.shape[0]
                filtered_deck[x,y] -= to_remove
                self.n_cards_in_filtered_deck -=1
                continue 
            #if we got here we know nothing of the card, just remove a total of 1 card from all the possible cards
            x,y = self.mask.nonzero()
            
            if x.shape[0] == 0:
                #some old debug prints
                print("---PRE EXCEPTION---")
                print(f"deck:\n{self.deck}\nhint:\n{hint}\nmask:\n{self.mask}\nplayer h_tracker:\n{main_player.hint_tracker}\ni:{i}\nrandomMask:\n{self.randomMask}")
                targets = np.logical_and(self.deck>=1, self.randomMask <= -1)
                assert np.any(targets)
                x,y = targets.nonzero()

                #adjust deck, cards in game and mask
                self.deck[x,y] -=1
                self.cards_in_game[x,y]-=1
                xr, yr = kn.nonzero()
                xr = xr[0]
                yr = yr[0]

                self.deck[xr,yr] += 1
                self.cards_in_game[xr,yr] += 1
                filtered_deck[xr,yr] = 0
                self.n_cards_in_filtered_deck-=1
                continue

            to_remove = 1/x.shape[0]

            filtered_deck[x,y] -= to_remove
            self.n_cards_in_filtered_deck -=1
        
        self.filtered_deck = filtered_deck

    def replace_randomicity(self,m):
        x,y = m.nonzero()

        #assert np.any(self.randomMask[x,y] >=1)
        targets = np.logical_and(self.deck>=1, self.randomMask<=-1)
        assert np.any(targets)
        xr,yr = targets.nonzero()
        
        index = randint(0, xr.shape[0])
        xr = xr[index]
        yr = yr[index]
        self.randomMask[x,y] =0

        self.deck[xr,yr]-=1
        self.cards_in_game[xr,yr]-=1
        self.randomMask[xr,yr]+=2


