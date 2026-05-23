package ontology.effects.binary;

import core.vgdl.VGDLSprite;
import core.content.InteractionContent;
import core.game.Game;
import core.logging.Logger;
import core.logging.Message;
import ontology.effects.Effect;

/**
 * Kills sprite1 when sprite2 is to the right of sprite1 and moving left (horizontal ricochet into the
 * player; horizontal analogue of KillIfFromAbove).
 */
public class KillIfFromRight extends Effect
{

    public KillIfFromRight(InteractionContent cnt)
    {
        is_kill_effect = true;
        this.parseParameters(cnt);
    }

    @Override
    public void execute(VGDLSprite sprite1, VGDLSprite sprite2, Game game)
    {
	if(sprite1 == null || sprite2 == null){
	    Logger.getInstance().addMessage(new Message(Message.WARNING, "Neither the 1st nor the 2nd sprite can be EOS with KillIfFromRight interaction."));
	    return;
	}

        double avatarCenterX = sprite1.lastrect.getMinX() + (sprite1.rect.width / 2.0);
        double shotCenterX = sprite2.lastrect.getMinX() + (sprite2.rect.width / 2.0);
        boolean shotRightOfAvatar = shotCenterX > avatarCenterX;
        boolean goingLeft = sprite2.rect.getMinX() < sprite2.lastrect.getMinX();

        applyScore=false;
        if (shotRightOfAvatar && goingLeft){
            applyScore=true;
            game.killSprite(sprite1, false);
        }
    }
}
