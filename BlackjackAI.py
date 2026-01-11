import copy
import random
import pygame
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# --- INITIALIZATION ---
pygame.init()
WIDTH, HEIGHT = 600, 900
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption('AI Blackjack Trainer')
timer = pygame.time.Clock()
fps = 60
font = pygame.font.Font('freesansbold.ttf', 44)
smaller_font = pygame.font.Font('freesansbold.ttf', 36)

# --- GAME CONSTANTS ---
cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
one_deck = 4 * cards
decks = 4

# --- AI VARIABLES ---
q_table = {}
learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0  # Start fully random
epsilon_decay = 0.9995

# --- HELPER FUNCTIONS ---
def calculate_score(hand):
    hand_score = 0
    aces_count = hand.count('A')
    for card in hand:
        if card in ['10', 'J', 'Q', 'K']:
            hand_score += 10
        elif card == 'A':
            hand_score += 11
        else:
            hand_score += int(card)
    while hand_score > 21 and aces_count > 0:
        hand_score -= 10
        aces_count -= 1
    return hand_score

def deal_cards(current_hand, current_deck):
    card = current_deck.pop(random.randint(0, len(current_deck) - 1))
    current_hand.append(card)
    return current_hand, current_deck

# --- AI LOGIC ---
def get_state(my_hand, dealer_hand):
    p_sum = calculate_score(my_hand)
    # Use the dealer's visible card (index 1)
    upcard = dealer_hand[1]
    d_val = 10 if upcard in ['J', 'Q', 'K'] else (11 if upcard == 'A' else int(upcard))
    usable_ace = 'A' in my_hand and calculate_score(my_hand) <= 21
    return (p_sum, d_val, usable_ace)

def choose_action(state):
    if state not in q_table:
        q_table[state] = [0.0, 0.0]
    if random.random() < epsilon:
        return random.randint(0, 1)
    return np.argmax(q_table[state])

def update_q_table(state, action, reward):
    if state not in q_table:
        q_table[state] = [0.0, 0.0]
    old_value = q_table[state][action]
    # Simple Temporal Difference update
    q_table[state][action] = old_value + learning_rate * (reward - old_value)

def visualize_policy():
    player_range = range(4, 22)
    dealer_range = range(2, 12)
    grid = np.zeros((len(player_range), len(dealer_range)))
    for i, p_sum in enumerate(player_range):
        for j, d_card in enumerate(dealer_range):
            state = (p_sum, d_card, False)
            if state in q_table:
                grid[i, j] = np.argmax(q_table[state])
    plt.figure(figsize=(10, 8))
    sns.heatmap(grid, annot=True, xticklabels=dealer_range, yticklabels=player_range, cmap="RdYlGn")
    plt.title("AI Strategy (0=Stand, 1=Hit)")
    plt.show()

# --- DRAWING FUNCTIONS ---
def draw_cards(player, dealer, reveal):
    for i, card in enumerate(player):
        pygame.draw.rect(screen, 'white', [70 + (70 * i), 460 + (5 * i), 120, 220], 0, 5)
        screen.blit(font.render(card, True, 'black'), (75 + 70 * i, 465 + 5 * i))
    for i, card in enumerate(dealer):
        pygame.draw.rect(screen, 'white', [70 + (70 * i), 160 + (5 * i), 120, 220], 0, 5)
        if i != 0 or reveal:
            screen.blit(font.render(card, True, 'black'), (75 + 70 * i, 165 + 5 * i))
        else:
            screen.blit(font.render('???', True, 'black'), (75 + 70 * i, 165 + 5 * i))

def draw_game(record):
    score_text = smaller_font.render(f'Wins: {record[0]}  Losses: {record[1]}  Draws: {record[2]}', True, 'white')
    screen.blit(score_text, (15, 840))

# --- MAIN LOOP ---
def main():
    global epsilon
    records = [0, 0, 0]
    game_deck = copy.deepcopy(decks * one_deck)
    my_hand, dealer_hand = [], []
    active = True
    initial_deal = True
    hand_active = True
    reveal_dealer = False
    outcome = 0 # 0: None, 1: Bust, 2: Win, 3: Loss, 4: Push

    iteration = 1
    run = True
    while run:
        pygame.time.delay(1)
        screen.fill('black')
        iter_text = smaller_font.render(f'Episode: {iteration}', True, 'lightblue')
        screen.blit(iter_text, (15, 780))
        
        if initial_deal:
            my_hand, dealer_hand = [], []
            for _ in range(2):
                my_hand, game_deck = deal_cards(my_hand, game_deck)
                dealer_hand, game_deck = deal_cards(dealer_hand, game_deck)
            initial_deal = False
            hand_active = True
            reveal_dealer = False
            outcome = 0

        # AI Turn
        if hand_active:
            state = get_state(my_hand, dealer_hand)
            action = choose_action(state)
            if action == 1: # Hit
                my_hand, game_deck = deal_cards(my_hand, game_deck)
                if calculate_score(my_hand) > 21:
                    hand_active = False
                    reveal_dealer = True
            else: # Stand
                hand_active = False
                reveal_dealer = True

        # Dealer Turn
        if not hand_active and reveal_dealer and outcome == 0:
            d_score = calculate_score(dealer_hand)
            p_score = calculate_score(my_hand)
            if d_score < 17 and p_score <= 21:
                dealer_hand, game_deck = deal_cards(dealer_hand, game_deck)
            else:
                # Calculate Outcome
                if p_score > 21: outcome = 1
                elif d_score > 21 or p_score > d_score: outcome = 2
                elif p_score < d_score: outcome = 3
                else: outcome = 4
                
                # Reward and Update
                reward = 1.0 if outcome == 2 else (-1.0 if outcome in [1, 3] else 0.0)
                update_q_table(state, action, reward)
                
                # Update Records
                if outcome == 2: records[0] += 1
                elif outcome in [1, 3]: records[1] += 1
                else: records[2] += 1

        # Drawing
        draw_cards(my_hand, dealer_hand, reveal_dealer)
        draw_game(records)

        if outcome != 0:
            iteration += 1
            game_deck = copy.deepcopy(decks * one_deck)
            msg = ["", "BUST!", "WIN!", "LOSE!", "PUSH!"][outcome]
            screen.blit(font.render(msg, True, 'green' if outcome == 2 else 'red'), (250, 25))
            initial_deal = True
            epsilon = max(0.01, epsilon * epsilon_decay)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v: # Press 'V' to see the heatmap
                    visualize_policy()

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()