package tracks.singlePlayer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

import core.logging.Logger;
import tools.Utils;
import tracks.ArcadeMachine;

/**
 * Created with IntelliJ IDEA. User: Diego Date: 04/10/13 Time: 16:29 This is a
 * Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class Test {

    public static void main(String[] args) {

		// Available tracks:
		String sampleRandomController = "tracks.singlePlayer.simple.sampleRandom.Agent";
		String doNothingController = "tracks.singlePlayer.simple.doNothing.Agent";
		String sampleOneStepController = "tracks.singlePlayer.simple.sampleonesteplookahead.Agent";
		String sampleFlatMCTSController = "tracks.singlePlayer.simple.greedyTreeSearch.Agent";

		String sampleMCTSController = "tracks.singlePlayer.advanced.sampleMCTS.Agent";
        String sampleRSController = "tracks.singlePlayer.advanced.sampleRS.Agent";
        String sampleRHEAController = "tracks.singlePlayer.advanced.sampleRHEA.Agent";
		String sampleOLETSController = "tracks.singlePlayer.advanced.olets.Agent";

		//Game settings
		boolean visuals = true;
		int seed = new Random().nextInt();

		// Game and level to play
		String game;
		String level1;
		if (args != null && args.length >= 2) {
			// e.g. java ... tracks.singlePlayer.Test examples/gridphysics/aliens_rules_ricochet.txt examples/gridphysics/aliens_lvl0.txt
			game = args[0];
			level1 = args[1];
		} else if (args != null && args.length == 1) {
			game = args[0];
			level1 = defaultLevelForOneArgGame(game);
		} else {
			//Load available games
			String spGamesCollection = "examples/all_games_sp.csv";
			String[][] games = Utils.readGames(spGamesCollection);
			int gameIdx = 0;
			int levelIdx = 0; // level names from 0 to 4 (game_lvlN.txt).
			String gameName = games[gameIdx][1];
			game = games[gameIdx][0];
			level1 = game.replace(gameName, gameName + "_lvl" + levelIdx);
		}

		String recordActionsFile = null;// "actions_" + games[gameIdx] + "_lvl"
						// + levelIdx + "_" + seed + ".txt";
						// where to record the actions
						// executed. null if not to save.

		// 1. This starts a game, in a level, played by a human.
		ArcadeMachine.playOneGame(game, level1, recordActionsFile, seed);

		// 2. This plays a game in a level by the controller.
//		ArcadeMachine.runOneGame(game, level1, visuals, sampleRHEAController, recordActionsFile, seed, 0);


		// 3. This replays a game from an action file previously recorded
	//	 String readActionsFile = recordActionsFile;
	//	 ArcadeMachine.replayGame(game, level1, visuals, readActionsFile);

		// 4. This plays a single game, in N levels, M times :
//		String level2 = new String(game).replace(gameName, gameName + "_lvl" + 1);
//		int M = 10;
//		for(int i=0; i<games.length; i++){
//			game = games[i][0];
//			gameName = games[i][1];
//			level1 = game.replace(gameName, gameName + "_lvl" + levelIdx);
//			ArcadeMachine.runGames(game, new String[]{level1}, M, sampleMCTSController, null);
//		}

		//5. This plays N games, in the first L levels, M times each. Actions to file optional (set saveActions to true).
//		int N = games.length, L = 2, M = 1;
//		boolean saveActions = false;
//		String[] levels = new String[L];
//		String[] actionFiles = new String[L*M];
//		for(int i = 0; i < N; ++i)
//		{
//			int actionIdx = 0;
//			game = games[i][0];
//			gameName = games[i][1];
//			for(int j = 0; j < L; ++j){
//				levels[j] = game.replace(gameName, gameName + "_lvl" + j);
//				if(saveActions) for(int k = 0; k < M; ++k)
//				actionFiles[actionIdx++] = "actions_game_" + i + "_level_" + j + "_" + k + ".txt";
//			}
//			ArcadeMachine.runGames(game, levels, M, sampleRHEAController, saveActions? actionFiles:null);
//		}


    }

	/**
	 * When only the game file is passed, pick a stock level whose {@code LevelMapping}
	 * matches that game family (under {@code examples/gridphysics/}). Aliens variants
	 * share aliens tiles; chopper / waves use their own level alphabets — do not load
	 * {@code aliens_lvl0.txt} for those or every cell logs "not defined" and the game breaks.
	 */
	public static String defaultLevelForOneArgGame(String gamePath) {
		String g = gamePath.replace('\\', '/').toLowerCase();
		// Prefer sibling "<game>_lvl0.txt" when present (works for data_collection variants).
		if (gamePath != null && gamePath.toLowerCase().endsWith(".txt")) {
			Path game = Paths.get(gamePath);
			Path parent = game.getParent();
			String file = game.getFileName() != null ? game.getFileName().toString() : "";
			if (!file.isEmpty()) {
				String stem = file.substring(0, file.length() - 4);
				Path sibling = (parent == null ? Paths.get(stem + "_lvl0.txt") : parent.resolve(stem + "_lvl0.txt"));
				if (Files.exists(sibling))
					return sibling.toString().replace('\\', '/');
				int rulesIdx = stem.indexOf("_rules_");
				if (rulesIdx > 0) {
					String baseStem = stem.substring(0, rulesIdx);
					Path baseSibling = (parent == null ? Paths.get(baseStem + "_lvl0.txt")
							: parent.resolve(baseStem + "_lvl0.txt"));
					if (Files.exists(baseSibling))
						return baseSibling.toString().replace('\\', '/');
				}
			}
		}
		if (g.contains("chopper"))
			return "examples/gridphysics/chopper_lvl0.txt";
		if (g.contains("waves"))
			return "examples/gridphysics/waves_lvl0.txt";
		if (g.contains("aliens"))
			return "examples/gridphysics/aliens_lvl0.txt";
		if (g.contains("eggomania"))
			return "examples/gridphysics/eggomania_lvl0.txt";
		if (g.contains("ikaruga"))
			return "examples/gridphysics/ikaruga_lvl0.txt";
		return "examples/gridphysics/aliens_lvl0.txt";
	}
}
