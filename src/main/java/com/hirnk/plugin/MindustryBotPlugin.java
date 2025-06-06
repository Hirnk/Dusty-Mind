package com.hirnk.plugin;

import arc.Events;
import arc.graphics.Color;
import mindustry.Vars;
import mindustry.core.NetServer;
import mindustry.game.EventType;
import mindustry.gen.Player;
import mindustry.mod.Plugin;
import mindustry.net.NetConnection;

public class MindustryBotPlugin extends Plugin {

    private static final String SERVER_IP = "127.0.0.1";
    private static final int SERVER_PORT = 7766;

    @Override
    public void init() {
        Events.on(EventType.PlayEvent.class, e -> {

            Vars.state.rules.waves = false;
            Vars.state.rules.infiniteResources = true;
        });
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
