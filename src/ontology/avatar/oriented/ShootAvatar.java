package ontology.avatar.oriented;

import java.awt.Dimension;
import java.util.ArrayList;
import java.util.Arrays;

import core.vgdl.VGDLRegistry;
import core.vgdl.VGDLSprite;
import core.content.SpriteContent;
import core.game.Game;
import ontology.Types;
import tools.Direction;
import tools.Utils;
import tools.Vector2d;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 22/10/13
 * Time: 18:10
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class ShootAvatar extends OrientedAvatar
{
    static int MAX_WEAPONS = 5;

    //This is the resource I need, to be able to shoot.
    public String ammo; //If ammo is null, no resource needed to shoot.
    public String[] ammos;
    public int[] ammoId;


    //This is the sprite I shoot
    public String stype;
    public String[] stypes;
    public int[] itype;

    /**
     * If true (default), new missiles copy the avatar's facing direction. If false, missiles keep the
     * orientation defined on the missile type (same behavior as FlakAvatar),
     * which is required for games where shots must travel along a fixed axis regardless of movement.
     */
    public boolean alignShotToOrientation = true;

    /** If true, USE fires every comma-separated weapon in {@code stype} in one tick (default: first only). */
    public boolean fireAllWeapons = false;

    /** Horizontal spacing between simultaneous shots when {@link #fireAllWeapons} is true; 0 = same cell. */
    public int spreadPixels = 10;

    public ShootAvatar(){}

    public ShootAvatar(Vector2d position, Dimension size, SpriteContent cnt)
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
        ammo = null;
        ammos = new String[MAX_WEAPONS];
        ammoId = new int[MAX_WEAPONS];
        stype = null;
        stypes = new String[MAX_WEAPONS];
        itype = new int[MAX_WEAPONS];
    }

    /**
     * This update call is for the game tick() loop.
     * @param game current state of the game.
     */
    public void updateAvatar(Game game, boolean requestInput, boolean[] actionMask)
    {
        super.updateAvatar(game, requestInput, actionMask);
        if(lastMovementType == Types.MOVEMENT.STILL)
            updateUse(game);
    }

    public void updateUse(Game game)
    {
        if(Utils.processUseKey(getKeyHandler().getMask(), getPlayerID())) {
            if (fireAllWeapons) {
                for (int i = 0; i < itype.length; i++) {
                    if (!hasAmmo(i))
                        return;
                }
                for (int i = 0; i < itype.length; i++)
                    shoot(game, i);
            } else {
                for (int i = 0; i < itype.length; i++) {
                    if (hasAmmo(i)) {
                        shoot(game, i);
                        break; // shoots the first priority weapon only
                    }
                }
            }
        }
    }

    protected void shoot(Game game, int idx)
    {
        Vector2d spawn;
        if (alignShotToOrientation) {
            Vector2d dir = this.orientation.getVector();
            dir.normalise();
            spawn = new Vector2d(this.rect.x + dir.x * this.lastrect.width,
                    this.rect.y + dir.y * this.lastrect.height);
        } else {
            spawn = new Vector2d(this.rect.x, this.rect.y);
        }

        if (fireAllWeapons && spreadPixels != 0 && itype != null && itype.length > 1) {
            double mid = (itype.length - 1) / 2.0;
            spawn.x += (int) Math.round((idx - mid) * spreadPixels);
        }

        VGDLSprite newOne = game.addSprite(itype[idx], spawn);

        if(newOne != null)
        {
            if(alignShotToOrientation && newOne.is_oriented) {
                Vector2d dir = this.orientation.getVector();
                dir.normalise();
                newOne.orientation = new Direction(dir.x, dir.y);
            }
            reduceAmmo(idx);
            newOne.setFromAvatar(true);
        }
    }

    /**
     * Maps a weapon index to an ammo row when {@code ammo} lists fewer types than {@code stype}
     * (e.g. one {@code bullet} pool for a triple shot): all weapons use slot 0.
     */
    protected int ammoSlotForWeaponIndex(int idx) {
        if (ammo == null || ammos == null || ammos.length == 0)
            return 0;
        if (idx >= 0 && idx < ammos.length)
            return idx;
        return 0;
    }

    protected boolean hasAmmo(int idx) {
        if (ammo == null)
            return true;

        int slot = ammoSlotForWeaponIndex(idx);
        if (slot < 0 || slot >= ammoId.length)
            return false;

        return resources.containsKey(ammoId[slot]) && resources.get(ammoId[slot]) > 0;
    }

    protected void reduceAmmo(int idx)
    {
        if (ammo == null)
            return;
        // One shared ammo string for multishot: subtract once (first projectile only).
        if (fireAllWeapons && ammos != null && ammos.length == 1 && idx > 0)
            return;

        int slot = ammoSlotForWeaponIndex(idx);
        if (slot < ammoId.length && resources.containsKey(ammoId[slot]))
            resources.put(ammoId[slot], resources.get(ammoId[slot]) - 1);
    }

    public void postProcess()
    {
        //Define actions here first.
        if(actions.size()==0)
        {
            actions.add(Types.ACTIONS.ACTION_USE);
            actions.add(Types.ACTIONS.ACTION_LEFT);
            actions.add(Types.ACTIONS.ACTION_RIGHT);
            actions.add(Types.ACTIONS.ACTION_DOWN);
            actions.add(Types.ACTIONS.ACTION_UP);
        }

        super.postProcess();

        stypes = stype.split(",");
        itype = new int[stypes.length];

        for (int i = 0; i < itype.length; i++)
            itype[i] = VGDLRegistry.GetInstance().getRegisteredSpriteValue(stypes[i]);
        if(ammo != null) {
            ammos = ammo.split(",");
            ammoId = new int[ammos.length];
            for (int i = 0; i < ammos.length; i++) {
                ammoId[i] = VGDLRegistry.GetInstance().getRegisteredSpriteValue(ammos[i]);
            }
        }
    }

    public VGDLSprite copy()
    {
        ShootAvatar newSprite = new ShootAvatar();
        this.copyTo(newSprite);
        return newSprite;
    }

    public void copyTo(VGDLSprite target)
    {
        ShootAvatar targetSprite = (ShootAvatar) target;
        targetSprite.stype = this.stype;
        targetSprite.itype = this.itype.clone();
        targetSprite.stypes = this.stypes.clone();
        targetSprite.ammo = this.ammo;
        targetSprite.ammoId= this.ammoId.clone();
        targetSprite.ammos = this.ammos.clone();
        targetSprite.alignShotToOrientation = this.alignShotToOrientation;
        targetSprite.fireAllWeapons = this.fireAllWeapons;
        targetSprite.spreadPixels = this.spreadPixels;

        super.copyTo(targetSprite);
    }
    
    @Override
    public ArrayList<String> getDependentSprites(){
    	ArrayList<String> result = new ArrayList<String>();
        if(ammo != null) result.addAll(Arrays.asList(ammos));
    	if(stype != null) result.addAll(Arrays.asList(stypes));
    	
    	return result;
    }
}
