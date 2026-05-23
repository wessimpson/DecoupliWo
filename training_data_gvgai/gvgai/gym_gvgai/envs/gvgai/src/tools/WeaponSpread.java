package tools;

import ontology.Types;

/**
 * Shared +/- angular spread for simultaneous weapon shots.
 */
public final class WeaponSpread {

    private WeaponSpread() {}

    public static boolean usesAngularSpread(boolean fireAllWeapons, int weaponCount, double spreadDegrees) {
        if (!fireAllWeapons || weaponCount < 2)
            return false;
        if (spreadDegrees > 0)
            return true;
        return weaponCount == 3;
    }

    public static double spreadRadians(double spreadDegrees) {
        if (spreadDegrees > 0)
            return Math.toRadians(spreadDegrees);
        return Math.toRadians(45.0);
    }

    public static Direction direction(int idx, int weaponCount, Direction facing, double spreadDegrees) {
        Vector2d base = facing.getVector().copy();
        if (Math.abs(base.x) + Math.abs(base.y) < 1e-6)
            base = new Vector2d(Types.DUP.x(), Types.DUP.y());
        else
            base.normalise();

        double baseAngle = Math.atan2(base.y, base.x);
        double mid = (weaponCount - 1) / 2.0;
        double angle = baseAngle + (idx - mid) * spreadRadians(spreadDegrees);
        return new Direction(Math.cos(angle), Math.sin(angle));
    }
}
