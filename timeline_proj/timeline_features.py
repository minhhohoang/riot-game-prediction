"""Utilities for converting Riot match and timeline JSON into minute-level features."""

import pandas as pd


def process_match(match_data, timeline_data):

    # --------------------------------------------------
    # Match-level information
    # --------------------------------------------------
    match_id = match_data["metadata"]["matchId"]
    game_version = match_data["info"]["gameVersion"]
    game_duration = match_data["info"]["gameDuration"]

    participants = match_data["info"]["participants"]
    frames = timeline_data["info"]["frames"]


    # --------------------------------------------------
    # Identify Blue and Red participants
    # --------------------------------------------------
    blue_ids = [
        p["participantId"]
        for p in participants
        if p["teamId"] == 100
    ]

    red_ids = [
        p["participantId"]
        for p in participants
        if p["teamId"] == 200
    ]


    # --------------------------------------------------
    # Determine winner
    # --------------------------------------------------
    blue_win = int(
        any(
            p["win"]
            for p in participants
            if p["teamId"] == 100
        )
    )


    # --------------------------------------------------
    # Cumulative event counters
    # --------------------------------------------------
    blue_kills = 0
    red_kills = 0

    blue_towers_destroyed = 0
    red_towers_destroyed = 0

    blue_dragons = 0
    red_dragons = 0

    blue_heralds = 0
    red_heralds = 0

    blue_barons = 0
    red_barons = 0

    blue_elders = 0
    red_elders = 0


    # --------------------------------------------------
    # Respawning structure state
    # --------------------------------------------------
    INHIB_RESPAWN_MS = 5 * 60 * 1000
    NEXUS_TURRET_RESPAWN_MS = 3 * 60 * 1000

    blue_inhib_respawns = []
    red_inhib_respawns = []

    blue_nexus_respawns = []
    red_nexus_respawns = []


    # --------------------------------------------------
    # Output rows
    # --------------------------------------------------
    rows = []


    # --------------------------------------------------
    # Process every timeline frame
    # --------------------------------------------------
    for frame in frames:

        current_time = frame["timestamp"]
        minute = round(current_time / 60000)

        pf = frame["participantFrames"]


        # ----------------------------------------------
        # Remove structures whose respawn timer finished
        # ----------------------------------------------
        blue_inhib_respawns = [
            t for t in blue_inhib_respawns
            if t > current_time
        ]

        red_inhib_respawns = [
            t for t in red_inhib_respawns
            if t > current_time
        ]

        blue_nexus_respawns = [
            t for t in blue_nexus_respawns
            if t > current_time
        ]

        red_nexus_respawns = [
            t for t in red_nexus_respawns
            if t > current_time
        ]


        # ----------------------------------------------
        # Process events in this frame
        # ----------------------------------------------
        for event in frame["events"]:

            event_type = event["type"]


            # ==========================================
            # Champion kills
            # ==========================================
            if event_type == "CHAMPION_KILL":

                killer_id = event.get("killerId", 0)

                if killer_id in blue_ids:
                    blue_kills += 1

                elif killer_id in red_ids:
                    red_kills += 1


            # ==========================================
            # Buildings
            # ==========================================
            elif event_type == "BUILDING_KILL":

                team_id = event.get("teamId")
                building_type = event.get("buildingType")
                tower_type = event.get("towerType")
                event_time = event["timestamp"]


                # --------------------------------------
                # Inhibitor
                # --------------------------------------
                if building_type == "INHIBITOR_BUILDING":

                    # Blue inhibitor destroyed
                    if team_id == 100:
                        blue_inhib_respawns.append(
                            event_time + INHIB_RESPAWN_MS
                        )

                    # Red inhibitor destroyed
                    elif team_id == 200:
                        red_inhib_respawns.append(
                            event_time + INHIB_RESPAWN_MS
                        )


                # --------------------------------------
                # Tower
                # --------------------------------------
                elif building_type == "TOWER_BUILDING":

                    # Nexus turret
                    if tower_type == "NEXUS_TURRET":

                        # Blue Nexus turret destroyed
                        if team_id == 100:
                            blue_nexus_respawns.append(
                                event_time + NEXUS_TURRET_RESPAWN_MS
                            )

                        # Red Nexus turret destroyed
                        elif team_id == 200:
                            red_nexus_respawns.append(
                                event_time + NEXUS_TURRET_RESPAWN_MS
                            )

                    # Normal tower
                    else:

                        # Red tower died -> Blue destroyed it
                        if team_id == 200:
                            blue_towers_destroyed += 1

                        # Blue tower died -> Red destroyed it
                        elif team_id == 100:
                            red_towers_destroyed += 1


            # ==========================================
            # Neutral objectives
            # ==========================================
            elif event_type == "ELITE_MONSTER_KILL":

                killer_team = event.get("killerTeamId")
                monster = event.get("monsterType")
                monster_subtype = event.get("monsterSubType")


                # --------------------------------------
                # Dragons
                # --------------------------------------
                if monster == "DRAGON":

                    # Elder Dragon
                    if monster_subtype == "ELDER_DRAGON":

                        if killer_team == 100:
                            blue_elders += 1

                        elif killer_team == 200:
                            red_elders += 1


                    # Normal elemental dragon
                    else:

                        if killer_team == 100:
                            blue_dragons += 1

                        elif killer_team == 200:
                            red_dragons += 1


                # --------------------------------------
                # Rift Herald
                # --------------------------------------
                elif monster == "RIFTHERALD":

                    if killer_team == 100:
                        blue_heralds += 1

                    elif killer_team == 200:
                        red_heralds += 1


                # --------------------------------------
                # Baron
                # --------------------------------------
                elif monster == "BARON_NASHOR":

                    if killer_team == 100:
                        blue_barons += 1

                    elif killer_team == 200:
                        red_barons += 1


        # --------------------------------------------------
        # Current structures alive
        # --------------------------------------------------
        blue_inhibs_alive = max(
            0,
            3 - len(blue_inhib_respawns)
        )

        red_inhibs_alive = max(
            0,
            3 - len(red_inhib_respawns)
        )

        blue_nexus_turrets_alive = max(
            0,
            2 - len(blue_nexus_respawns)
        )

        red_nexus_turrets_alive = max(
            0,
            2 - len(red_nexus_respawns)
        )


        # --------------------------------------------------
        # Team totals from participant frames
        # --------------------------------------------------
        blue_gold = 0
        red_gold = 0

        blue_xp = 0
        red_xp = 0

        blue_cs = 0
        red_cs = 0


        for pid in blue_ids:

            player = pf[str(pid)]

            blue_gold += player["totalGold"]
            blue_xp += player["xp"]

            blue_cs += (
                player["minionsKilled"]
                + player["jungleMinionsKilled"]
            )


        for pid in red_ids:

            player = pf[str(pid)]

            red_gold += player["totalGold"]
            red_xp += player["xp"]

            red_cs += (
                player["minionsKilled"]
                + player["jungleMinionsKilled"]
            )


        # --------------------------------------------------
        # Save current game state
        # --------------------------------------------------
        rows.append({

            # Match metadata
            "match_id": match_id,
            "minute": minute,
            "game_version": game_version,
            "game_duration": game_duration,

            # Economy
            "blue_gold": blue_gold,
            "red_gold": red_gold,
            "gold_diff": blue_gold - red_gold,

            "blue_xp": blue_xp,
            "red_xp": red_xp,
            "xp_diff": blue_xp - red_xp,

            "blue_cs": blue_cs,
            "red_cs": red_cs,
            "cs_diff": blue_cs - red_cs,


            # Combat
            "blue_kills": blue_kills,
            "red_kills": red_kills,
            "kill_diff": blue_kills - red_kills,


            # Towers
            "blue_towers_destroyed": blue_towers_destroyed,
            "red_towers_destroyed": red_towers_destroyed,
            "tower_diff": (
                blue_towers_destroyed
                - red_towers_destroyed
            ),


            # Dragons
            "blue_dragons": blue_dragons,
            "red_dragons": red_dragons,

            "dragon_diff": (
                blue_dragons
                - red_dragons
            ),

            "blue_dragons_to_soul": max(
                0,
                4 - blue_dragons
            ),

            "red_dragons_to_soul": max(
                0,
                4 - red_dragons
            ),

            "blue_has_soul": int(
                blue_dragons >= 4
            ),

            "red_has_soul": int(
                red_dragons >= 4
            ),


            # Elder
            "blue_elders": blue_elders,
            "red_elders": red_elders,


            # Herald
            "blue_heralds": blue_heralds,
            "red_heralds": red_heralds,


            # Baron
            "blue_barons": blue_barons,
            "red_barons": red_barons,


            # Current base state
            "blue_inhibs_alive": blue_inhibs_alive,
            "red_inhibs_alive": red_inhibs_alive,

            "blue_nexus_turrets_alive":
                blue_nexus_turrets_alive,

            "red_nexus_turrets_alive":
                red_nexus_turrets_alive,


            # Target
            "blue_win": blue_win
        })


    return pd.DataFrame(rows)