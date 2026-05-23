package ontology.avatar;

import java.awt.Dimension;
import java.util.ArrayList;

import core.vgdl.VGDLRegistry;
import core.vgdl.VGDLSprite;
import core.content.SpriteContent;
import core.game.Game;
import ontology.Types;
import tools.Utils;
import tools.Vector2d;

/**
 * Created with IntelliJ IDEA.
 * User: Diego
 * Date: 22/10/13
 * Time: 18:08
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class FlakAvatar extends HorizontalAvatar
{
    public String stype;
    public String[] stypes;

    public int itype;
    public int[] itypes;

    public boolean fireAllWeapons;
    public int spreadPixels;

    //This is the resource I need, to be able to shoot.
    public String ammo; //If ammo is null, no resource needed to shoot.
    public int ammoId;
    public int minAmmo; //-1 if not used. minimum amount of ammo needed for shooting.
    public int ammoCost; //1 if not used. amount of ammo to subtract after shooting once.


    public FlakAvatar(){}

    public FlakAvatar(Vector2d position, Dimension size, SpriteContent cnt)
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
        ammoId = -1;
        minAmmo = -1;
        ammoCost = 1;
        fireAllWeapons = false;
        spreadPixels = 10;
        color = Types.GREEN;
    }


    public void postProcess()
    {
        //Define actions here first.
        if(actions.size()==0)
        {
            actions.add(Types.ACTIONS.ACTION_USE);
            actions.add(Types.ACTIONS.ACTION_LEFT);
            actions.add(Types.ACTIONS.ACTION_RIGHT);
        }

        super.postProcess();

        stypes = stype.split(",");
        itypes = new int[stypes.length];
        for(int i = 0; i < itypes.length; i++)
        {
            stypes[i] = stypes[i].trim();
            itypes[i] =  VGDLRegistry.GetInstance().getRegisteredSpriteValue(stypes[i]);
            if(itypes[i] < 0)
                throw new IllegalArgumentException("Undefined sprite " + stypes[i] + " in FlakAvatar stype=" + stype);
        }
        itype = itypes[0];
        if(ammo != null)
            ammoId = VGDLRegistry.GetInstance().getRegisteredSpriteValue(ammo);
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
        if(Utils.processUseKey(getKeyHandler().getMask(), getPlayerID()) && hasAmmo()) //use primary set of keys, idx = 0
        {
            int weaponCount = fireAllWeapons ? itypes.length : 1;
            boolean addedAny = false;
            for(int i = 0; i < weaponCount; i++)
            {
                Vector2d spawn = new Vector2d(this.rect.x, this.rect.y);
                if(fireAllWeapons && spreadPixels != 0 && weaponCount > 1)
                {
                    double mid = (weaponCount - 1) / 2.0;
                    spawn.x += (int) Math.round((i - mid) * spreadPixels);
                }

                VGDLSprite added = game.addSprite(itypes[i], spawn);
                if(added != null){ //singleton sprites could not add anything here.
                    addedAny = true;
                    added.setFromAvatar(true);
                }
            }
            if(addedAny) {
                reduceAmmo();
            }
        }
    }

    private boolean hasAmmo()
    {
        if(ammo == null)
            return true; //no ammo defined, I can shoot.

        //If I have ammo, I must have enough resource of ammo type to be able to shoot.
        if(resources.containsKey(ammoId))
            if(minAmmo > -1)
                return resources.get(ammoId) > minAmmo;
            else
                return resources.get(ammoId) > 0;

        return false;
    }

    private void reduceAmmo()
    {
        if(ammo != null && resources.containsKey(ammoId))
        {
            resources.put(ammoId, resources.get(ammoId) - ammoCost);
        }
    }
    public VGDLSprite copy()
    {
        FlakAvatar newSprite = new FlakAvatar();
        this.copyTo(newSprite);
        return newSprite;
    }

    public void copyTo(VGDLSprite target)
    {
        FlakAvatar targetSprite = (FlakAvatar) target;
        targetSprite.stype = this.stype;
        targetSprite.stypes = this.stypes == null ? null : this.stypes.clone();
        targetSprite.itype= this.itype;
        targetSprite.itypes= this.itypes == null ? null : this.itypes.clone();
        targetSprite.ammo = this.ammo;
        targetSprite.ammoId= this.ammoId;
        targetSprite.ammoCost = this.ammoCost;
        targetSprite.minAmmo= this.minAmmo;
        targetSprite.fireAllWeapons = this.fireAllWeapons;
        targetSprite.spreadPixels = this.spreadPixels;
        super.copyTo(targetSprite);
    }
    
    @Override
    public ArrayList<String> getDependentSprites(){
        ArrayList<String> result = new ArrayList<String>();
        if(ammo != null) result.add(ammo);
        if(stypes != null) {
            for(String type : stypes)
                result.add(type);
        } else if(stype != null) {
            result.add(stype);
        }

        return result;
    }
}
