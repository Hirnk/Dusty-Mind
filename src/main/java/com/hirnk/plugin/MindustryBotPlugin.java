package com.hirnk.plugin;

import arc.Events;
import arc.graphics.Color;
import arc.util.Log;
import mindustry.Vars;
import mindustry.content.Blocks;
import mindustry.content.Items;
import mindustry.content.UnitTypes;
import mindustry.core.NetClient;
import mindustry.core.NetServer;
import mindustry.entities.units.BuildPlan;
import mindustry.game.EventType;
import mindustry.game.Team;
import mindustry.gen.Player;
import mindustry.mod.Plugin;
import mindustry.net.NetConnection;
import mindustry.world.Block;
import mindustry.world.Tile;
import mindustry.world.blocks.production.Drill;
import mindustry.world.blocks.distribution.Conveyor;
import mindustry.world.blocks.storage.CoreBlock;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicReference;

public class MindustryBotPlugin extends Plugin {

    private static final String SERVER_IP = "127.0.0.1";
    private static final int SERVER_PORT = 7766;

    private DataOutputStream out;
    private Thread client;
    private final boolean running = true;

    // Single bot reference
    private final AtomicReference<Player> bot = new AtomicReference<>();

    @Override
    public void init() {
        Events.on(EventType.PlayEvent.class, e -> {
            startClient();

            Player newBot = createBot("RL-Agent", Color.sky, "RL-IP");
            bot.set(newBot);

            Vars.state.rules.waves = false;
            Vars.state.rules.infiniteResources = true;
            Vars.state.rules.buildSpeedMultiplier = 100f;
            UnitTypes.alpha.buildRange = 2500f;
        });
    }

    private void startClient() {
        client = new Thread(() -> {
            try (Socket socket = new Socket(SERVER_IP, SERVER_PORT);
                 DataInputStream in = new DataInputStream(socket.getInputStream());
                 DataOutputStream dos = new DataOutputStream(socket.getOutputStream())) {

                out = dos;

                Log.info("Connected to Python server");

                // Send first observation immediately after connecting
                byte[] obs = getObservation();
                out.writeInt(obs.length);
                out.write(obs);
                out.flush();

                Log.info("Sent initial observation to server");

                while (running && !socket.isClosed()) {
                    byte[] lenBytes = in.readNBytes(4);
                    int len = ByteBuffer.wrap(lenBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();

                    Log.info("Receiving action: expected length = " + len);

                    byte[] msg = in.readNBytes(len);
                    if (msg.length != len) {
                        Log.err("Received incomplete action: " + msg.length + "/" + len);
                        return;
                    }
                    Log.info("Received complete action");
                    processAction(msg);

                    obs = getObservation();
                    out.writeInt(obs.length);
                    out.write(obs);
                    out.flush();
                }
            } catch (IOException e) {
                Log.err("Socket error: " + e.getMessage());
            }
        });
        client.start();
    }

    private void processAction(byte[] data) {
        Player p = bot.get();
        if (p == null || p.unit() == null) {
            Log.err("Bot or its unit is null. Skipping action.");
            return;
        }

        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int type = buf.getInt();
        int x = buf.getInt();
        int y = buf.getInt();
        int rot = buf.getInt();

        Block b = switch (type) {
            case 0 -> Blocks.mechanicalDrill;
            case 1 -> Blocks.conveyor;
            default -> null;
        };

        if (b == null) {
            Log.err("Invalid block type received: " + type);
            return;
        }

        p.unit().plans().add(new BuildPlan(x, y, rot, b));
        NetClient.sendChatMessage(bot.get(), "Bot placed " + b.name + " at (" + x + ", " + y + ")");
    }

    private byte[] getObservation() {
        int w = 400, h = 400, l = 6;
        byte[][][] map = new byte[l][w][h];

        for (Tile t : Vars.world.tiles) {
            Block b = t.block();
            Block o = t.overlay();
            if (b instanceof CoreBlock) map[0][t.x][t.y] = 1;
            else if (b instanceof Drill) map[1][t.x][t.y] = 1;
            else if (b instanceof Conveyor) map[2][t.x][t.y] = 1;

            if (o.name.equals(Blocks.oreCopper.name)) map[3][t.x][t.y] = 1;

            if (t.solid()) map[4][t.x][t.y] = 1;

            map[5][t.x][t.y] = t.build != null ? (byte) t.build.rotation() : 0;
        }

        ByteBuffer buf = ByteBuffer.allocate(l * w * h + 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < l; i++)
            for (int x = 0; x < w; x++)
                for (int y = 0; y < h; y++)
                    buf.put(map[i][x][y]);

        buf.putFloat(Vars.state.teams.cores(Team.sharded).first().items.get(Items.copper));

        return buf.array();
    }

    private Player createBot(String name, Color color, String ip) {
        Player p = Player.create();
        p.name = name;
        p.color = color;
        p.con = new NetConnection(ip) {
            @Override public void send(Object o, boolean r) { }
            @Override public void close() { }
        };
        p.add();
        NetServer.connectConfirm(p);
        Events.fire(new EventType.PlayerConnect(p));
        return p;
    }
}
