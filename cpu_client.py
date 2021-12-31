#!/usr/bin/env python3

from sys import argv, stdout
from threading import Thread
import GameData
import socket
from constants import *
import os
import time

from solver.Move import Move
from solver.HanabiSolver import HanabiSolver
from solver.Naive.NSolver import NSolver

if len(argv) < 4:
    print("You need the player name to start the game.")
    #exit(-1)
    playerName = "s288265" # For debug
    ip = HOST
    port = PORT + 1
    solver_name = "NSolver"
    solver = {}
else:
    playerName = argv[3]
    ip = argv[1]
    port = int(argv[2])
    solver_name = "NSolver"
    solver = {}

run = True

statuses = ["Lobby", "Game", "GameHint"]
gameStatuses = ["NotReady","Ready","Start", "Update", "MyTurn", "OthersTurn"]


status = statuses[0]
gameStatus = gameStatuses[0]
hintState = ("", "")
players=[] #needed to store the order of players


def manage_solver():
    global run
    global status
    global gameStatus
    while run
        match gameStatus:
            case "NotReady":
                p = input("type anything to get ready")
                s.send(GameData.ClientPlayerStartRequest(playerName).serialize())
                gameStatus = gameStatuses[1]
                
            case "Ready":
                pass
            case "Start":
                print("asking for initial info")
                s.send(GameData.ClientGetGameStateRequest(playerName).serialize())
            case "OthersTurn":
                solver.Enforce()
            case "Update":
                print("requesting an update on player cards")
                s.send(GameData.ClientGetGameStateRequest(playerName).serialize())
            case "MyTurn":
                print("finding move")
                move = solver.FindMove()
                match move.type:
                    case 0:
                        print(f"discarding card {move.card_n}")
                        s.send(GameData.ClientPlayerDiscardCardRequest(playerName, move.card_n).serialize())
                    case 1:
                        print(f"playing card {move.card_n}")
                        s.send(GameData.ClientPlayerPlayCardRequest(playerName, move.card_n).serialize())
                    case 2:
                        print(f"hinting {move.h_player} for their {move.h_type} {move.h_value}")
                        s.send(GameData.ClientHintData(playerName, move.h_player, move.h_type, move.hvalue).serialize())
    os._exit(0)


    

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #global playerName
    request = GameData.ClientPlayerAddData(playerName)
    s.connect((HOST, PORT)) 
    s.send(request.serialize())
    data = s.recv(DATASIZE)
    data = GameData.GameData.deserialize(data)
    if type(data) is GameData.ServerPlayerConnectionOk:
        print("Connection accepted by the server. Welcome " + playerName)
    print("[" + playerName + " - " + status + "]: ", end="")
    Thread(target=manage_solver).start()
    while run:
        dataOk = False
        data = s.recv(DATASIZE)
        if not data:
            continue
        data = GameData.GameData.deserialize(data)
        if type(data) is GameData.ServerPlayerStartRequestAccepted:
            dataOk = True
            print("Ready: " + str(data.acceptedStartRequests) + "/"  + str(data.connectedPlayers) + " players")
            data = s.recv(DATASIZE)
            data = GameData.GameData.deserialize(data)
        if type(data) is GameData.ServerStartGameData:
            dataOk = True
            print("Game start!")
            #print(data.players)
            
            s.send(GameData.ClientPlayerReadyData(playerName).serialize())
            players=data.players
            status = statuses[1]
            gameStatus = gameStatuses[2]
        if type(data) is GameData.ServerGameStateData:
            dataOk = True
            print("Current player: " + data.currentPlayer)
            print("Player hands: ")
            for p in data.players:
                print(p.toClientString())
            print("Table cards: ")
            for pos in data.tableCards:
                print(pos + ": [ ")
                for c in data.tableCards[pos]:
                    print(c.toClientString() + " ")
                print("]")
            print("Discard pile: ")
            for c in data.discardPile:
                print("\t" + c.toClientString())            
            print("Note tokens used: " + str(data.usedNoteTokens) + "/8")
            print("Storm tokens used: " + str(data.usedStormTokens) + "/3")
            if gameStatus == "Start":
                #initialize solver
                match solver_name:
                    case "NSolver":
                        solver = NSolver(data, players,playerName)
            elif gameStatus == "Update":
                
                solver.RecordMove(data, "draw")

            #change state to "MyTurn" or "OthersTurn"
            if data.currentPlayer == playerName:
                gameStatus = gameStatuses[4]
            else:
                gameStatus = gameStatuses[5]
        

        if type(data) is GameData.ServerActionInvalid:
            dataOk = True
            print("Invalid action performed. Reason:")
            print(data.message)
        if type(data) is GameData.ServerActionValid:
            dataOk = True
            print("Action valid!")
            print("Current player: " + data.player)
            solver.RecordMove(data, "discard")
            gameStatus = gameStatuses[3]
        if type(data) is GameData.ServerPlayerMoveOk:
            dataOk = True
            print("Nice move!")
            print("Current player: " + data.player)
            solver.RecordMove(data, "play")
            gameStatus = gameStatuses[3]
        if type(data) is GameData.ServerPlayerThunderStrike:
            dataOk = True
            print("OH NO! The Gods are unhappy with you!")
            solver.RecordMove(data, "thunder")
            gameStatus = gameStatuses[3]
        if type(data) is GameData.ServerHintData:
            dataOk = True
            print("Hint type: " + data.type)
            print("Player " + data.destination + " cards with value " + str(data.value) + " are:")
            for i in data.positions:
                print("\t" + str(i))
            solver.RecordMove(data, "hint")
            gameStatus = gameStatuses[3]
        if type(data) is GameData.ServerInvalidDataReceived:
            dataOk = True
            print(data.data)
        if type(data) is GameData.ServerGameOver:
            dataOk = True
            print(data.message)
            print(data.score)
            print(data.scoreMessage)
            stdout.flush()
            run = False
        if not dataOk:
            print("Unknown or unimplemented data type: " +  str(type(data)))
        print("[" + playerName + " - " + status + "]: ", end="")
        stdout.flush()