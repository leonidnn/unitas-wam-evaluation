"""Extract object, contact and gripper diagnostics from a saved episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from replay_data import load_recording, restore_frame, sha256


def target_grasp_events(arrays, body_list, positions, rotations, contact_pairs, mug, plate):
    policy = np.flatnonzero(arrays['phases'] == 2)
    start = int(policy[0] - 1) if len(policy) else 0
    mug_position = positions[:, body_list.index(mug)]
    eef_index = body_list.index('gripper0_eef')
    eef_rotation = np.asarray(rotations)[:, eef_index].reshape(-1, 3, 3)
    relative = np.einsum('tij,ti->tj', eef_rotation, mug_position - positions[:, eef_index])
    lift = mug_position[:, 2] - mug_position[start, 2]
    bilateral, plate_contact, one_finger = [], [], []
    fingers = ({'gripper0_leftfinger', 'gripper0_finger_joint1_tip'},
               {'gripper0_rightfinger', 'gripper0_finger_joint2_tip'})
    for pairs in contact_pairs:
        others = {b if a == mug else a for a, b in pairs if mug in (a, b)}
        sides = [bool(others & finger) for finger in fingers]
        bilateral.append(all(sides))
        one_finger.append(any(sides))
        plate_contact.append(plate in others)

    def first(mask, begin=start + 1):
        indices = np.flatnonzero(np.asarray(mask)[begin:])
        return int(indices[0]) + begin if len(indices) else None

    def event(index):
        if index is None:
            return None
        return {'frame': index, 'policy_steps': index - start,
                'seconds_from_policy_start': float(arrays['times'][index] - arrays['times'][start])}

    closed = first(arrays['actions'][:, -1] > 0)
    touch = first(one_finger)
    held = first(bilateral)
    raised = first(lift >= 0.02)
    placed = first(plate_contact)
    loss = first(np.logical_not(one_finger), held + 1) if held is not None else None
    if placed is not None and loss is not None and loss >= placed:
        loss = None
    stop = min([i for i in (loss, placed, len(lift) - 1) if i is not None])
    relative_drift = (float(np.linalg.norm(relative[held:stop + 1] - relative[held], axis=1).max())
                      if held is not None and stop >= held else None)
    return {'first_close_command': event(closed), 'first_finger_contact': event(touch),
            'first_bilateral_contact': event(held), 'first_lift_at_least_20mm': event(raised),
            'first_total_finger_contact_loss_before_plate': event(loss),
            'first_plate_contact': event(placed), 'lift_threshold_m': 0.02,
            'max_mug_origin_drift_in_eef_frame_before_release_m': relative_drift}


def inspect(directory, output):
    model, data, metadata, arrays = load_recording(directory)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ''
             for i in range(model.nbody)]
    bodies = {name: i for i, name in enumerate(names)
              if name and ('gripper' in name or
                           (not name.startswith('robot0') and
                            (model.body_jntnum[i] > 0 or
                             any(part in name for part in ('mug', 'bowl', 'tray', 'plate',
                                                         'drawer', 'book', 'caddy')))))}
    if not bodies:
        raise ValueError('No object or gripper bodies in this task')
    mug_ids = {i for name, i in bodies.items() if 'porcelain_mug' in name}
    plate_ids = {i for name, i in bodies.items() if 'plate' in name}
    gripper_ids = {i for name, i in bodies.items() if 'gripper' in name}
    positions, rotations, contact_rows, contact_pairs = [], [], [], []
    articulated_ids = [i for i in range(model.njnt)
                       if model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_SLIDE,
                                               mujoco.mjtJoint.mjJNT_HINGE)
                       and not names[model.jnt_bodyid[i]].startswith(('robot0', 'gripper'))]
    articulated = []
    for index in range(len(arrays['times'])):
        restore_frame(model, data, metadata, arrays, index)
        positions.append([data.xpos[i].copy() for i in bodies.values()])
        rotations.append([data.xmat[i].copy() for i in bodies.values()])
        articulated.append(data.qpos[model.jnt_qposadr[articulated_ids]].copy())
        counts = {'mug_plate': 0, 'mug_gripper': 0, 'gripper_other': 0}
        pairs = set()
        for contact in data.contact[:data.ncon]:
            a, b = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if a != b and (a in bodies.values() or b in bodies.values()):
                pairs.add(tuple(sorted((names[a], names[b]))))
            if (a in mug_ids and b in plate_ids) or (b in mug_ids and a in plate_ids):
                counts['mug_plate'] += 1
            if (a in mug_ids and b in gripper_ids) or (b in mug_ids and a in gripper_ids):
                counts['mug_gripper'] += 1
            if ((a in gripper_ids and b not in gripper_ids | mug_ids)
                    or (b in gripper_ids and a not in gripper_ids | mug_ids)):
                counts['gripper_other'] += 1
        contact_rows.append(list(counts.values()))
        contact_pairs.append([list(pair) for pair in sorted(pairs)])
    positions = np.asarray(positions)
    contacts = np.asarray(contact_rows)
    body_list = list(bodies)
    mug_name = next((name for name in body_list if name == 'porcelain_mug_1_main'), None)
    if mug_name is None:
        mug_name = next((name for name in body_list if 'porcelain_mug' in name), None)
    plate_name = next((name for name in body_list if name == 'plate_1_main'), None)
    if plate_name is None:
        plate_name = next((name for name in body_list if 'plate' in name), None)
    summary = {'source': str(directory), 'result': metadata.get('result'),
               'inspector_sha256': sha256(__file__),
               'body_names': body_list, 'contact_columns': list(counts),
               'frames': len(positions), 'scope': 'State-derived observations, no causal attribution'}
    if mug_name and plate_name:
        mug = positions[:, body_list.index(mug_name)]
        plate = positions[:, body_list.index(plate_name)]
        delta = mug - plate
        policy = np.flatnonzero(arrays['phases'] == 2)
        start = int(policy[0] - 1) if len(policy) else 0
        summary.update(mug_body=mug_name, plate_body=plate_name,
                       mug_lift_max_m=float((mug[start:, 2] - mug[start, 2]).max()),
                       minimum_xy_distance_m=float(np.linalg.norm(delta[start:, :2], axis=1).min()),
                       final_xy_distance_m=float(np.linalg.norm(delta[-1, :2])),
                       final_mug_minus_plate_z_m=float(delta[-1, 2]))
    if mug_name and plate_name and 'gripper0_eef' in body_list:
        summary['target_grasp_events'] = target_grasp_events(
            arrays, body_list, positions, rotations, contact_pairs, mug_name, plate_name)
    for column, name in enumerate(counts):
        indices = np.flatnonzero(contacts[:, column])
        summary[name] = {'frames': len(indices),
                         'first_time_s': float(arrays['times'][indices[0]]) if len(indices) else None,
                         'last_time_s': float(arrays['times'][indices[-1]]) if len(indices) else None}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'diagnostics.npz', positions=positions, contacts=contacts,
                        rotations=np.asarray(rotations), articulated_qpos=np.asarray(articulated),
                        body_names=np.asarray(body_list), times=arrays['times'],
                        actions=arrays['actions'], phases=arrays['phases'])
    (output / 'contacts.json').write_text(json.dumps(contact_pairs), encoding='utf-8')
    summary['articulated_joint_names'] = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                                          for i in articulated_ids]
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspect(args.input, args.output)
