package ontology.sprites.producer;

import java.awt.Dimension;
import java.util.ArrayList;
import java.util.Arrays;

import core.vgdl.VGDLRegistry;
import core.vgdl.VGDLSprite;
import core.content.SpriteContent;
import core.game.Game;
import ontology.Types;
import tools.Direction;
import tools.Vector2d;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 21/10/13
 * Time: 18:24
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class SpawnPoint extends SpriteProducer
{
    public double prob;
    public int total;
    public int counter;
    public String stype;
    public String[] stypes;
    public int itype;
    public int[] itypes;
    public Direction spawnorientation;
    public boolean fireAllWeapons;
    public int spreadDegrees;

    protected int start;

    public SpawnPoint(){}

    public SpawnPoint(Vector2d position, Dimension size, SpriteContent cnt)
    {
        //Init the sprite
        this.init(position, size);

        //Specific class default parameter values.
        loadDefaults();

        //Parse the arguments.
        this.parseParameters(cnt);
    }

    protected void loadDefaults()
    {
        super.loadDefaults();
        prob = 1.0;
        total = 0;
        start = -1;
        color = Types.BLACK;
        cooldown = 1;
        is_static = true;
        spawnorientation = Types.DNONE;
        fireAllWeapons = false;
        spreadDegrees = 0;
        itype = -1;
        itypes = new int[] {-1};
    }

    public void postProcess()
    {
        super.postProcess();
        is_stochastic = (prob > 0 && prob < 1);
        counter = 0;
        if(stype != null) //Could be, if we're using different stype variants in subclasses.
        {
            stypes = stype.split(",");
            itypes = VGDLRegistry.GetInstance().explode(stype);
            itype = itypes.length > 0 ? itypes[0] : -1;
        }
    }

    protected boolean shouldSpawnThisTick(Game game)
    {
        if(start == -1)
            start = game.getGameTick();

        float rollDie = game.getRandomGenerator().nextFloat();
        return ((start+game.getGameTick()) % cooldown == 0) && rollDie < prob;
    }

    protected void spawnProjectile(Game game)
    {
        int[] spawnTypes = (itypes == null || itypes.length == 0) ? new int[] {itype} : itypes;
        int spawnCount = fireAllWeapons ? spawnTypes.length : 1;
        for (int i = 0; i < spawnCount; i++)
            spawnOne(game, spawnTypes[i], i, spawnCount);
    }

    public void update(Game game)
    {
        if (shouldSpawnThisTick(game)) {
            spawnProjectile(game);
            counter++;
        }

        super.update(game);

        if (total > 0 && counter >= total)
            game.killSprite(this, false);
    }

    protected void spawnOne(Game game, int spawnType, int index, int count)
    {
        if (spawnType < 0)
            return;

        VGDLSprite newSprite = game.addSprite(spawnType, this.getPosition());
        if(newSprite != null) {
            if(fireAllWeapons && count > 1 && spreadDegrees != 0) {
                Direction base = baseSpawnDirection();
                double mid = (count - 1) / 2.0;
                newSprite.orientation = rotateDirection(base, (index - mid) * spreadDegrees);
            }
            //We set the orientation given by default it this was passed.
            else if(!(spawnorientation.equals(Types.DNONE)))
                newSprite.orientation = spawnorientation.copy();
            //If no orientation given, we set the one from the spawner.
            else if (newSprite.orientation.equals(Types.DNONE))
                newSprite.orientation = this.orientation.copy();
        }
    }

    protected Direction baseSpawnDirection()
    {
        if(!(spawnorientation.equals(Types.DNONE)))
            return spawnorientation;
        if(!(this.orientation.equals(Types.DNONE)))
            return this.orientation;
        return Types.DRIGHT;
    }

    protected Direction rotateDirection(Direction base, double degrees)
    {
        double radians = Math.toRadians(degrees);
        double cos = Math.cos(radians);
        double sin = Math.sin(radians);
        double x = base.x();
        double y = base.y();
        double rx = x * cos - y * sin;
        double ry = x * sin + y * cos;
        double norm = Math.sqrt(rx * rx + ry * ry);
        if (norm > 0) {
            rx /= norm;
            ry /= norm;
        }
        return new Direction(rx, ry);
    }

    /**
     * Updates spawn itype with newitype
     * @param itype - current spawn type
     * @param newitype - new spawn type to replace the first
     */
    public void updateItype(int itype, int newitype) {
        this.itype = newitype;
    }

    public VGDLSprite copy()
    {
        SpawnPoint newSprite = new SpawnPoint();
        this.copyTo(newSprite);
        return newSprite;
    }

    public void copyTo(VGDLSprite target)
    {
        SpawnPoint targetSprite = (SpawnPoint) target;
        targetSprite.prob = this.prob;
        targetSprite.total = this.total;
        targetSprite.counter = this.counter;
        targetSprite.stype = this.stype;
        targetSprite.stypes = this.stypes == null ? null : this.stypes.clone();
        targetSprite.itype = this.itype;
        targetSprite.itypes = this.itypes == null ? null : this.itypes.clone();
        targetSprite.spawnorientation = this.spawnorientation.copy();
        targetSprite.fireAllWeapons = this.fireAllWeapons;
        targetSprite.spreadDegrees = this.spreadDegrees;
        targetSprite.start = this.start;
        super.copyTo(targetSprite);
    }

    @Override
    public ArrayList<String> getDependentSprites(){
    	ArrayList<String> result = new ArrayList<String>();
    	if(stypes != null) result.addAll(Arrays.asList(stypes));
        else if(stype != null) result.add(stype);
    	
    	return result;
    }
}
