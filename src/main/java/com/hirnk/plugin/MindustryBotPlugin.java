package com.hirnk.plugin;

import arc.Events;
import arc.graphics.Color;
import arc.util.Log;
import mindustry.Vars;
import mindustry.content.Blocks;
import mindustry.content.Items;
import mindustry.content.UnitTypes;
import mindustry.core.NetServer;
import mindustry.entities.units.BuildPlan;
import mindustry.game.EventType;
import mindustry.game.Team;
import mindustry.gen.Call;
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
    private static final int SERVER_PORT = 5555;

    private DataOutputStream out;
    private Thread client;
    private final boolean running = true;

    // Single bot reference
    private final AtomicReference<Player> bot = new AtomicReference<>();

    @Override
    public void init() {
        Events.on(EventType.ServerLoadEvent.class, e -> startClient());

        Events.on(EventType.PlayEvent.class, e -> {
            // Create and store the single bot once per game start
            if (bot.get() == null) {
                Player newBot = createBot("RL-Agent", Color.sky, "RL-IP");
                bot.set(newBot);
            }

            Vars.state.rules.waves = false;
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

                while (running && !socket.isClosed()) {
                    int len = in.readInt();
                    byte[] msg = new byte[len];
                    in.readFully(msg);

                    if (len == 16) processAction(msg);

                    byte[] obs = getObservation();
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
        if (p == null) return;

        ByteBuffer buf = ByteBuffer.wrap(data);
        int type = buf.getInt(), x = buf.getInt(), y = buf.getInt(), rot = buf.getInt();

        Block b = switch (type) {
            case 0 -> Blocks.mechanicalDrill;
            case 1 -> Blocks.conveyor;
            default -> null;
        };
        if (b == null) return;

        p.unit().plans().add(new BuildPlan(x, y, rot, b));
        Call.sendChatMessage("Bot placed " + type + " at " + x + "," + y);
    }

    private byte[] getObservation() {
        int w = 400, h = 400, l = 5;
        byte[][][] map = new byte[l][w][h];

        for (Tile t : Vars.world.tiles) {
            Block b = t.block();
            if (b instanceof CoreBlock) map[0][t.x][t.y] = 1;
            else if (b instanceof Drill) map[1][t.x][t.y] = 1;
            else if (b instanceof Conveyor) map[2][t.x][t.y] = 1;
            else if (t.solid()) map[3][t.x][t.y] = 1;

            map[4][t.x][t.y] = t.build != null ? (byte) t.build.rotation() : 0;
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
