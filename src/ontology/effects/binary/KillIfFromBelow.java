package ontology.effects.binary;

import core.vgdl.VGDLSprite;
import core.content.InteractionContent;
import core.game.Game;
import core.logging.Logger;
import core.logging.Message;
import ontology.effects.Effect;

/**
 * Kills sprite1 when sprite2 is below sprite1's vertical center and moving upward
 * (mirror of KillIfFromAbove for ricocheting shots in vertical chopper-style games).
 */
public class KillIfFromBelow extends Effect
{

    public KillIfFromBelow(InteractionContent cnt)
    {
        is_kill_effect = true;
        this.parseParameters(cnt);
    }

    @Override
    public void execute(VGDLSprite sprite1, VGDLSprite sprite2, Game game)
    {
	if(sprite1 == null || sprite2 == null){
	    Logger.getInstance().addMessage(new Message(Message.WARNING, "Neither the 1st nor the 2nd sprite can be EOS with KillIfFromBelow interaction."));
	    return;
	}

        double avatarCenterY = sprite1.lastrect.getMinY() + (sprite1.rect.height / 2.0);
        double bombCenterY = sprite2.lastrect.getMinY() + (sprite2.rect.height / 2.0);
        boolean otherLower = bombCenterY > avatarCenterY;
        boolean goingUp = sprite2.rect.getMinY() < sprite2.lastrect.getMinY();

        applyScore=false;
        if (otherLower && goingUp){
            applyScore=true;
            game.killSprite(sprite1, false);
        }
    }
}
