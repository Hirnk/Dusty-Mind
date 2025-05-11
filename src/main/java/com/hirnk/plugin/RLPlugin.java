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
import arc.struct.Seq;
import arc.util.*;
import mindustry.Vars;
import mindustry.core.NetServer;
import mindustry.game.EventType;
import mindustry.game.EventType.PlayerConnect;
import mindustry.game.Team;
import mindustry.gen.Player;
import mindustry.mod.*;
import mindustry.net.NetConnection;
import mindustry.type.Item;
import mindustry.world.Block;
import mindustry.world.blocks.storage.CoreBlock;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.net.*;

public class RLPlugin extends Plugin {

    private ServerSocket serverSocket;
    private int agentCounter;
    private final Seq<Block> tempSeq = new Seq<>();

    public RLPlugin() {
        startSocketServer();
    }

    private void startSocketServer() {
        new Thread(() -> {
            try {
                serverSocket = new ServerSocket(1337);
                Log.info("Socket server started on port 1337");

                while (true) {
                    Socket clientSocket = serverSocket.accept();
                    handleClient(clientSocket);
                }

            } catch (IOException e) {
                Log.err("Error starting socket server", e);
            }
        }).start();
    }

    private void handleClient(Socket socket) {
        new Thread(() -> {
            try (
                    BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()))
            ) {
                String input = reader.readLine();
                JSONObject request = new JSONObject(input);
                JSONObject response = new JSONObject();

                switch (request.getString("command")) {
                    case "get_blocks":
                        response.put("map", getBuildableBlocks());
                        break;
                    case "get_map":
                        response.put("map", getMapObservation());
                        break;
                    case "resources":
                        response.put("map", getCoreResources());
                        break;
                }

                writer.write(response.toString());
                writer.newLine();
                writer.flush();

            } catch (IOException e) {
                Log.err("Client communication error", e);
            }
        }).start();
    }

    @Override
    public void init() {
        Events.on(EventType.PlayEvent.class, e -> {
            createBot("RL-Agent #" + agentCounter, Color.sky, String.valueOf(agentCounter++));
            Log.info("Loaded Agent!");

            //temporal

            Vars.state.rules.waves = false;
            Vars.state.rules.buildSpeedMultiplier = 100f;
        });
    }

    public void addPlayer(Player p) {
        p.add();
        NetServer.connectConfirm(p);
        Events.fire( new PlayerConnect(p));
    }

    public void createBot(String name, Color color, String ip) {
        Player p = Player.create();

        p.name = name;
        p.color = color;
        p.con = new NetConnection(ip) {
            @Override
            public void send(Object object, boolean reliable) {}

            @Override
            public void close() {}
        };

        addPlayer(p);
    }

    @Override
    public void registerClientCommands(CommandHandler handler) {
        handler.register("bot","Create an agent.", arg -> createBot("RL-Agent #" + agentCounter, Color.sky, String.valueOf(agentCounter++)));
    }

    //observation design

    private JSONArray getMapObservation() {
        int height = Vars.world.height();
        int width = Vars.world.height();

        JSONArray rows = new JSONArray();

        for (int y = 0; y < height; y++) {
            JSONArray row = new JSONArray();
            for (int x = 0; x < width; x++) {
                var tile = Vars.world.tile(x, y);
                int blockId = 0;

                if (tile != null && tile.block() != null) {
                    blockId = tile.block().id;
                }

                row.put(blockId);
            }
            rows.put(row);
        }

        return rows;
    }

    //action design

    public JSONArray getBuildableBlocks() {
        JSONArray array = new JSONArray();
        tempSeq.selectFrom(Vars.content.blocks(), Block::isPlaceable);
        array.putAll(tempSeq);

        return array;
    }

    public JSONArray getCoreResources() {
        JSONArray array = new JSONArray();

        CoreBlock.CoreBuild core = Vars.state.teams.get(Team.sharded).core();

        if (core != null) {
            for (Item item : Vars.content.items()) {
                int amount = core.items.get(item);
                array.put(amount);
            }
        }

        return array;
    }
}
