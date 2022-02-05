from solver.Deck import Deck
from . import FPlayer
import numpy as np
from .. import utils

DECK_VERBOSE = False

class FDeck(Deck):
    def __init__(self, main_player: FPlayer, n_players: int):
        super().__init__()
        self.main_player = main_player
        self.n_players = n_players
        self.n_cards_in_deck = 50
        self.n_cards_in_filtered_deck = 50
        self.n_cards_in_game = 50
        self.cards_in_game = np.copy(self.deck) #this structure keeps track of all cards still in game (instead of just the ones in the deck)
        
        #keeps track of cards that can't be in deck or in agent's hand
        self.mask = np.zeros((5,5)) != 0

        #these are the EXPECTED playability/discardability of the cards present in the deck (only)
        #Can also be interpreted as the values for the card
        #you are drawing next. This has to be initialized after all players drew and whenever a play is made or an hint 
        #to agent happens with update_expected_values
        self.playability_rc = 0
        self.discardability_rc = 0

        #same as above, but for cards in agent's hand + deck
        self.playability_ad = 0
        self.discardability_ad = 0

        #this datastructure stores the cards that are in the deck only: in brief 
        #the agents hints are translated into decimal values that are subtracted from self.deck
        #initialized in update_expected_values 
        self.filtered_deck = np.zeros((5,5), dtype = np.float32)

        #this is supposed to keep track for each color of the cards that will be useless for the rest of the game
        #i.e. a red 5 is useless if red firework is at value 3 and all red 4 are out of the game (initialized in update_death_line)
        self.death_line = np.zeros(5, dtype=np.int16) + 5

    def update_expected_values(self, fireworks: np.ndarray):
        needed_values = fireworks +1
        needed_colors = np.array([i for i in range(0,5)])

        needed_colors= needed_colors[needed_values<5]
        needed_values= needed_values[needed_values<5]
        
        self.update_death_line(fireworks)

        #get filtered deck (an estimate of the cards in the deck computed with agent's hints)
        filtered_deck = np.array(self.deck, dtype=np.float32)
        self.n_cards_in_filtered_deck = self.n_cards_in_deck
        for i in range(0, self.main_player.cardsInHand):
            if self.n_cards_in_filtered_deck <= 0:
                break

            hint = self.main_player.hints[:,:, i]
            
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
                x = x[mask_filter]
                y = y[mask_filter]

                to_remove = 1/x.shape[0]
                filtered_deck[x,y] -= to_remove
                self.n_cards_in_filtered_deck -=1
                continue 
            #if we got here we know nothing of the card, just remove a total of 1 card from all the possible cards
            x,y = self.mask.nonzero()
            to_remove = 0
            to_remove = 1/x.shape[0]

            filtered_deck[x,y] -= to_remove
            self.n_cards_in_filtered_deck -=1

        #cleanup (the above code can produce negative quantities for cards since mask is not updated among cycle iterations)
        #distribute negative values among positive ones
        neg = filtered_deck < 0
        if np.any(neg):
            #print("--------CLEANING UP---------")
            neg_x, neg_y = neg.nonzero()
            pos_x, pos_y = (filtered_deck>0).nonzero()

            #print(f"negative: {neg_x}, {neg_y} positive: {pos_x}, {pos_y}")
            #print(filtered_deck)

            to_add = -np.sum(filtered_deck[neg_x,neg_y])
            if pos_x.shape[0] > 0:
                to_add = to_add/pos_x.shape[0]
            else:
                to_add = 0
            
            #print(f"to_add: {to_add}")
            filtered_deck[pos_x,pos_y]+= to_add
            filtered_deck[neg_x,neg_y] = 0

        self.filtered_deck = filtered_deck

        #compute a priori playability and discardability for cards in agent's hand + deck
        n_playable = np.sum(self.deck[needed_values, needed_colors])
        self.playability_ad = n_playable/self.n_cards_in_deck

        tot_discardability = 0
        arange = np.arange(5)
        #for each color
        for i in range(0,needed_colors.shape[0]):
            discardable_indexes = arange<needed_values[i]
            dead_indexes = arange >= self.death_line[i]
            non_trivial_indexes = np.logical_not(np.logical_or(discardable_indexes,dead_indexes))
            #trivial: cards that are lower than firework of their color have discardability 1
            tot_discardability += np.sum(self.deck[discardable_indexes, i])

            #trivial: cards that are beyond the death line
            tot_discardability += np.sum(self.deck[dead_indexes, i])

            #for the rest, compute their discardability through formula at the end of evaluate_known_cards()
            non_trivials = self.deck[non_trivial_indexes, i]

            #discardability of each card * number of cards of that kind
            nt_discardabilities = self.discardability_fn(self.cards_in_game[non_trivial_indexes, i])*non_trivials
                        #(1- np.reciprocal(self.cards_in_game[arange>=needed_values[i], i]))*non_trivials
            tot_discardability += np.sum(nt_discardabilities)
        
        self.discardability_ad = tot_discardability/self.n_cards_in_deck

        if self.n_cards_in_filtered_deck <= 0:
            return 

        #if we're not in endgame, compute the values also for cards in deck only

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

            #discardability of each card * number of cards of that kind
            nt_discardabilities = self.discardability_fn(self.cards_in_game[non_trivial_indexes, i])*non_trivials
                    #1- np.reciprocal(self.cards_in_game[arange>=needed_values[i], i]))*non_trivials
            tot_discardability += np.sum(nt_discardabilities)
        
        self.discardability_rc = tot_discardability/self.n_cards_in_filtered_deck

    def update_death_line(self, fireworks: np.ndarray):
        self.death_line = np.zeros(5,dtype=np.int16) + 5
        mask = self.cards_in_game<=0
        #for each color
        for i in range(0,5):
            if np.any(mask[fireworks[i]:, i]):

                #death line[i] is the minimum card value greater than the firework for which there are no more cards left
                self.death_line[i] = np.argmax(mask[fireworks[i]:, i]) + fireworks[i]


    def RemoveCards(self, to_remove: np.ndarray):
        #a set of cards were removed from the deck (drawn) and we know them. update 
        self.deck[to_remove[0,:],to_remove[1,:]]-=1
        self.mask = self.deck > 0
        self.n_cards_in_deck -= to_remove.shape[1]

    def RemoveCardsFromGame(self, to_remove: np.ndarray):
        self.cards_in_game[to_remove[0,:],to_remove[1,:]]-=1
        self.n_cards_in_game -= to_remove.shape[1]
    

    def evaluate_known_cards(self, to_evaluate: np.ndarray, fireworks: np.ndarray):
        card_colors = to_evaluate[1, :]
        relevant_needed_nums = fireworks[card_colors] + 1 
        relevant_death_line = self.death_line[card_colors]

        #a known card has playability max only if it is immediately playable. Minimum playability otherwise
        playabilities = np.zeros(to_evaluate.shape[1], dtype= np.float32)
        playable_cards = to_evaluate[0, :] == relevant_needed_nums
        playabilities[playable_cards]+=1

        #if the given card has a lower value than current firework value of its color, it's very discardable
        discardabilities = np.zeros(to_evaluate.shape[1], dtype= np.float32)
        discardable_cards = to_evaluate[0, :] <relevant_needed_nums

        dead_cards = to_evaluate[0,:] >= relevant_death_line

        discardable_cards = np.logical_or(discardable_cards, dead_cards)

        discardabilities[discardable_cards] += 1

        non_discardables = np.logical_not(discardable_cards)

        #fetch number of cards in game that are equal to the one we want to evaluate
        discardabilities[non_discardables] = self.cards_in_game[to_evaluate[0, non_discardables], to_evaluate[1, non_discardables]]
        #discardability is 0 if the card is the only one in game nad proportional to the number of other equal cards present
        #formula is subject to change
        discardabilities[non_discardables] = self.discardability_fn(discardabilities[non_discardables])

        #TODO: give max discardability to cards that can't be played anymore (ex. a red 5 if all red 4 are in discard pile) (DONE)

        return playabilities, discardabilities

    def evaluate_unknown_cards(self, n_cards: int, fireworks: np.ndarray, hints: np.ndarray):
        needed_nums = fireworks + 1 
        needed_colors = np.arange(5)

        #we give deck's expected value by default
        playabilities = np.zeros(n_cards, dtype= np.float32) + self.playability_rc
        discardabilities = np.zeros(n_cards, dtype= np.float32) + self.discardability_rc

        for i in range(0, n_cards):
            hint = hints[:,:, i]
            
            #perfect knowledge case
            kn = hint >= 2
            if np.any(kn):
                x, y = kn.nonzero()
                if needed_nums[y[0]] == x[0]:
                    playabilities[i] = 1
                else:
                    playabilities[i] = 0
                

                if needed_nums[y[0]] > x[0] or x[0] >= self.death_line[y[0]]:
                    #trivial discardability
                    discardabilities[i] = 1
                else:
                    #same formula as for known cards
                    n_equal_cards = self.cards_in_game[x[0],y[0]]
                    discardabilities[i] = self.discardability_fn(n_equal_cards)
            
            #partial knowledge case
            kn = hint > 0
            if np.any(kn):
                #mean of everything this can be
                x,y = kn.nonzero()

                mask_filter = self.mask[x, y]
                x = x[mask_filter]
                y = y[mask_filter]

                playabilities[i] = np.mean(needed_nums[y] == x)

                #trivial
                tmp_disc = np.zeros(x.shape[0], dtype= np.float32)
                tr_i = needed_nums[y]>x
                tr_d = x >= self.death_line[y]
                tr_i = np.logical_or(tr_i,tr_d)
                tmp_disc[tr_i] = 1
                
                #non trivial
                nt_i = np.logical_not(tr_i)
                tmp_disc[nt_i] = self.cards_in_game[x[nt_i], y[nt_i]]
                tmp_disc[nt_i] = self.discardability_fn(tmp_disc[nt_i])
                discardabilities[i] = np.mean(tmp_disc)
            
            #negative knowledge case
            kn = hint < 0
            if np.any(kn):
                kn = np.logical_not(kn)
                x,y = kn.nonzero()

                mask_filter = self.mask[x, y]
                x = x[mask_filter]
                y = y[mask_filter]

                playabilities[i] = np.mean(needed_nums[y] == x)

                #trivial
                tmp_disc = np.zeros(x.shape[0], dtype= np.float32)
                tr_i = needed_nums[y]>x
                tr_d = x >= self.death_line[y]
                tr_i = np.logical_or(tr_i,tr_d)
                tmp_disc[tr_i] = 1
                
                #non trivial
                nt_i = np.logical_not(tr_i)
                tmp_disc[nt_i] = self.cards_in_game[x[nt_i], y[nt_i]]
                tmp_disc[nt_i] = self.discardability_fn(tmp_disc[nt_i])
                discardabilities[i] = np.mean(tmp_disc)
            
            #we know nothing of this card, values are defaults, move on to the next
        
        return playabilities, discardabilities

    def DeckToString(self):
        if DECK_VERBOSE:
            string = (f"Deck (cards: {self.n_cards_in_deck}):\n{self.deck}\nFiltered Deck (cards: {self.n_cards_in_filtered_deck}):\n{self.filtered_deck}\n"
                        f"Cards in game (cards: {self.n_cards_in_game}):\n{self.cards_in_game}\nDeck values:\n" 
                        f"playability: {self.playability_rc} discardability: {self.discardability_rc}\n"
                        f"Deck + agent values:\n"
                        f"playability: {self.playability_ad} discardability: {self.discardability_ad}")
        else:
            string = f"Deck (cards: {self.n_cards_in_deck})\n{self.deck}\ncards in game (cards: {self.n_cards_in_game}\n{self.cards_in_game}"
        return string

    def discardability_fn(self, n_cards_in_game):
        if type(n_cards_in_game) == np.ndarray:
            return np.power(1- np.reciprocal(n_cards_in_game), 2)
        elif type(n_cards_in_game) == int:
            assert n_cards_in_game > 0
            return np.power(1-(1/n_cards_in_game))

    def update_filtered_deck(self, main_player: FPlayer):
        #get filtered deck (an estimate of the cards in the deck computed with agent's hints)
        filtered_deck = np.array(self.deck, dtype=np.float32)
        self.n_cards_in_filtered_deck = self.n_cards_in_deck
        for i in range(0, main_player.cardsInHand):
            if self.n_cards_in_filtered_deck <= 0:
                break

            hint = self.main_player.hints[:,:, i]
            
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
                x = x[mask_filter]
                y = y[mask_filter]

                to_remove = 1/x.shape[0]
                filtered_deck[x,y] -= to_remove
                self.n_cards_in_filtered_deck -=1
                continue 
            #if we got here we know nothing of the card, just remove a total of 1 card from all the possible cards
            x,y = self.mask.nonzero()
            to_remove = 0
            to_remove = 1/x.shape[0]

            filtered_deck[x,y] -= to_remove
            self.n_cards_in_filtered_deck -=1
        





                
                
    
    
        




        