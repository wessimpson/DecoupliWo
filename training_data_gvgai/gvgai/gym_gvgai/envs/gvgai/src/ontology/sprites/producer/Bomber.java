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
import tools.WeaponSpread;

/**
 * Spawning NPC (e.g. aliens). Supports {@code fireAllWeapons=True} with comma-separated
 * {@code stype} for triple shots at +/-45° around {@link #spawnorientation} (aim), not movement {@link #orientation}.
 */
public class Bomber extends SpawnPoint
{
    public boolean fireAllWeapons = false;
    public double spreadDegrees = 0;

    public String[] stypes;
    public int[] itypes;

    public Bomber(){}

    public Bomber(Vector2d position, Dimension size, SpriteContent cnt)
    {
        this.init(position, size);
        loadDefaults();
        this.parseParameters(cnt);
    }

    protected void loadDefaults()
    {
        super.loadDefaults();
        is_static = false;
        is_oriented = true;
        orientation = Types.DRIGHT.copy();
        is_npc = true;
    }

    @Override
    public void postProcess()
    {
        if (fireAllWeapons && stype != null && stype.contains(",")) {
            stypes = stype.split(",");
            itypes = new int[stypes.length];
            for (int i = 0; i < stypes.length; i++)
                itypes[i] = VGDLRegistry.GetInstance().getRegisteredSpriteValue(stypes[i].trim());
            if (itypes.length > 0)
                itype = itypes[0];
            counter = 0;
            is_stochastic = (prob > 0 && prob < 1);
            loadImage();
            return;
        }
        stypes = null;
        itypes = null;
        super.postProcess();
    }

    protected boolean usesAngularSpread() {
        return itypes != null
                && WeaponSpread.usesAngularSpread(fireAllWeapons, itypes.length, spreadDegrees);
    }


    @Override
    protected void spawnProjectile(Game game)
    {
        if (itypes != null && itypes.length >= 3 && fireAllWeapons)
            spawnSpread(game);
        else if (usesAngularSpread())
            spawnSpread(game);
        else
            super.spawnProjectile(game);
    }

    protected Vector2d spawnPositionForShot(Game game, Direction shotDir)
    {
        int bs = game.getBlockSize();
        Vector2d dir = shotDir.getVector();
        dir.normalise();
        double x = this.rect.getCenterX() + dir.x * bs - bs / 2.0;
        double y = this.rect.getCenterY() + dir.y * bs - bs / 2.0;
        if (x < 0)
            x = this.getPosition().x;
        if (y < 0)
            y = this.getPosition().y;
        return new Vector2d(x, y);
    }

    /** Aim direction for spawned missiles (spawnorientation), separate from movement facing. */
    protected Direction shootFacing()
    {
        if (spawnorientation != null && !spawnorientation.equals(Types.DNONE))
            return spawnorientation.copy();
        return this.orientation.copy();
    }

    protected void spawnSpread(Game game)
    {
        int n = itypes.length;
        Direction facing = shootFacing();
        for (int i = 0; i < n; i++) {
            Direction shotDir = WeaponSpread.direction(i, n, facing, spreadDegrees);
            Vector2d pos = spawnPositionForShot(game, shotDir);
            // force=true: same stype repeated (bomb,bomb,bomb) must bypass singleton limits.
            VGDLSprite newSprite = game.addSprite(itypes[i], pos, true);
            if (newSprite != null && newSprite.is_oriented)
                newSprite.orientation = shotDir.copy();
        }
    }

    public VGDLSprite copy()
    {
        Bomber newSprite = new Bomber();
        this.copyTo(newSprite);
        return newSprite;
    }

    public void copyTo(VGDLSprite target)
    {
        Bomber targetSprite = (Bomber) target;
        targetSprite.fireAllWeapons = this.fireAllWeapons;
        targetSprite.spreadDegrees = this.spreadDegrees;
        if (this.stypes != null)
            targetSprite.stypes = this.stypes.clone();
        if (this.itypes != null)
            targetSprite.itypes = this.itypes.clone();
        super.copyTo(targetSprite);
    }

    @Override
    public ArrayList<String> getDependentSprites() {
        ArrayList<String> result = new ArrayList<String>();
        if (stypes != null)
            result.addAll(Arrays.asList(stypes));
        else if (stype != null)
            result.add(stype);
        return result;
    }
}
