/*
 * This file is part of TemplatePlugin. A template plugin for Mindustry to get you started quickly.
 *
 * MIT License
 *
 * Copyright (c) 2024 Xpdustry
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package com.hirnk.plugin;
import arc.Events;
import arc.graphics.Color;
import arc.util.*;
import mindustry.Vars;
import mindustry.content.Blocks;
import mindustry.content.Items;
import mindustry.content.UnitTypes;
import mindustry.core.NetServer;
import mindustry.entities.units.BuildPlan;
import mindustry.game.EventType;
import mindustry.game.EventType.PlayerConnect;
import mindustry.game.Team;
import mindustry.gen.Call;
import mindustry.gen.Player;
import mindustry.mod.*;
import mindustry.net.NetConnection;
import mindustry.world.Block;
import mindustry.world.Tile;
import mindustry.world.blocks.distribution.Conveyor;
import mindustry.world.blocks.production.Drill;
import mindustry.world.blocks.storage.CoreBlock;

import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicReference;

public class MindustryBotPlugin extends Plugin {

    int socketPort = 7777;

    private ServerSocket serverSocket;
    private Thread serverThread;
    private volatile boolean running = true;
    private AtomicReference<Player> botPlayer = new AtomicReference<>();

    private int agentCounter;

    @Override
    public void init() {

        // Create the bot player once the server is ready

        Events.on(EventType.PlayEvent.class, e -> {
            startSocketServer(socketPort);

            createBot("RL-Agent #" + agentCounter, Color.sky, String.valueOf(agentCounter++));
            Log.info("Loaded Agent!");
            botPlayer.set(createBot());

            //temporal

            Vars.state.rules.waves = false;
            Vars.state.rules.buildSpeedMultiplier = 100f;
            UnitTypes.alpha.buildRange = 2500f;
        });
    }

    private void startSocketServer(int port) {
        serverThread = new Thread(() -> {
            try {
                serverSocket = new ServerSocket(port);
                Log.info("Java socket server listening on port " + port);

                while (running) {
                    try (Socket clientSocket = serverSocket.accept()) {
                        System.out.println("Java client connected");

                        DataInputStream in = new DataInputStream(clientSocket.getInputStream());
                        DataOutputStream out = new DataOutputStream(clientSocket.getOutputStream());

                        while (running && !clientSocket.isClosed()) {
                            // receive action data from client
                            int actionLen = in.readInt();
                            byte[] actionBytes = new byte[actionLen];
                            in.readFully(actionBytes);

                            processAction(actionBytes);

                            // send observation data back to client
                            byte[] obsBytes = getObservationBytes();
                            out.writeInt(obsBytes.length);
                            out.write(obsBytes);
                            out.flush();
                        }
                    } catch (IOException e) {
                        System.err.println("Client disconnected or error: " + e.getMessage());
                    }
                }
            } catch (IOException e) {
                System.err.println("Socket server error: " + e.getMessage());
            }
        });
        serverThread.setDaemon(true);
        serverThread.start();
    }

    // Agent - Action

    private void processAction(byte[] actionBytes) {
        // Example: actionBytes could encode tile x,y and action type
        // Decode it (e.g. ByteBuffer), then run commands to place blocks or issue commands

        if (botPlayer.get() == null) return;

        ByteBuffer buffer = ByteBuffer.wrap(actionBytes);
        int actionType = buffer.getInt();  // e.g., 0=build drill,1=build conveyor
        int tileX = buffer.getInt();
        int tileY = buffer.getInt();
        int rotation = buffer.getInt();

        Block block;

        switch (actionType) {
            case 0 -> block = Blocks.mechanicalDrill;
            case 1 -> block = Blocks.conveyor;

            default -> {
                return;
            }
        }

        Player bot = botPlayer.get();
        if (bot == null || block == null) return;

        BuildPlan plan = new BuildPlan(tileX, tileY, rotation, block);

        bot.unit().plans().add(plan);

        Call.sendChatMessage("Bot placing action " + actionType + " at " + tileX + "," + tileY);
    }

    private byte[] getObservationBytes() {
        int width = 400, height = 400, layers = 4;

        // one-hot encoding
        byte[][][] map = new byte[layers][width][height];

        for (Tile t : Vars.world.tiles) {
            Block b = t.block();

            if (b instanceof CoreBlock) map[0][t.x][t.y] = 1;
            else if (b instanceof Drill) map[1][t.x][t.y] = 1;
            else if (b instanceof Conveyor) map[2][t.x][t.y] = 1;
            else if (t.solid()) map[3][t.x][t.y] = 1;
        }

        ByteBuffer buffer = ByteBuffer.allocate(layers * width * height + 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        for (int l = 0; l < layers; l++) {
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    buffer.put(map[l][x][y]);
                }
            }
        }

        // Add core copper amount
        float copper = Vars.state.teams.cores(Team.sharded).first().items.get(Items.copper);
        buffer.putFloat(copper);

        return buffer.array();
    }

    public void addPlayer(Player p) {
        p.add();
        NetServer.connectConfirm(p);
        Events.fire(new PlayerConnect(p));
    }

    public Player createBot() {
        return createBot("Agent#" + agentCounter, Color.sky, String.valueOf(agentCounter++));
    }

    public Player createBot(String name, Color color, String ip) {
        Player p = Player.create();

        p.name = name;
        p.color = color;
        p.con = new NetConnection(ip) {
            @Override
            public void send(Object object, boolean reliable) {
            }

            @Override
            public void close() {
            }
        };

        addPlayer(p);

        return p;
    }

    @Override
    public void registerClientCommands(CommandHandler handler) {
        handler.<Player>register("bot", "Create an agent.", arg -> createBot("RL-Agent #" + agentCounter, Color.sky, String.valueOf(agentCounter++)));
        handler.<Player>register("endRL", "Stop the bot socket server", args -> {
            running = false;
            try {
                if (serverSocket != null) serverSocket.close();
            } catch (IOException e) {
                System.err.println("Client disconnected or error: " + e.getMessage());
            }
            Call.sendChatMessage("Bot server stopped.");
        });
    }
}

class BotAction {
    public final Player bot;
    public final Block block;
    public final int tileX, tileY;
    public final int rotation;

    public BotAction(Player bot, Block block, int tileX, int tileY, int rotation) {
        this.bot = bot;
        this.block = block;
        this.tileX = tileX;
        this.tileY = tileY;
        this.rotation = rotation;
    }
}