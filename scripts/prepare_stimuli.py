"""Prepare metadata-only nutrient controls and global food-group folds.

Reads an explicit allowlist of metadata columns, never human ratings.
Pairing is a conservative operational construction, not a validated human judgment.
"""
import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.fft import dctn
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    'cake': '1 6 13 28 29 36 39 42 51 55 59 74 84 88 89 95 96 97 99 100 103 106 107 122 127 130 133 135 136 139 146 151 162 163 179 331 353 354 355 357 419 420 421',
    'muffin': '14 48 78 80 90 93 134 137 138',
    'croissant': '7 15 66 114 120 184 187 351 443 474',
    'donut_pastry': '18 19 20 40 41 50 101 119 126 169 178 374 375 376 377 400 422 441 475',
    'waffle_pancake': '9 16 105 470 471 483 488 507',
    'bread_roll': '62 156 292 300 352 366 423 425 426 439 440 469 480 495 549',
    'dry_cereal_crispbread': '193 226 244 348 372',
    'milk_cereal': '227 349',
    'burger_hotdog': '2 45 52 53 65 87 91 318 326 473 547',
    'burger_fries_meal': '3 17 86 485',
    'sandwich': '12 23 34 57 58 69 72 75 77 487 512 527',
    'bread_cottage_cheese': '205 213 225 240',
    'doner': '68 188',
    'pizza': '32 61 81 85 108 131 403 489',
    'pasta_sauced': '10 33 37 73 76 141 142 143 144 145 310 378 383 384 385 521 551',
    'potato_fried': '22 46 60 306 317 363',
    'potato_soft_side': '304 316 327 490',
    'meat_potato_meal': '30 47 314 322 337 484 486 493 556 565',
    'sushi': '279 308 369 371 564',
    'fish_vegetable_meal': '323 406 408',
    'fresh_salad': '196 201 212 215 219 228 229 232 242 332 482 526',
    'tomato_mozzarella': '566 567',
    'cookies': '4 5 31 44 79 121 148 158 164 171 177 289 298 503 504 505 506 523 533',
    'chocolate_confection': '35 83 111 112 159 160 161 165 166 167 168 170 172 173 174 286 287 290 291 293 295 296 297 344 465 499 500 501 510 511',
    'sugar_candy': '21 94 102 123 124 125 128 147 153 154 157 336 338 339 340',
    'icecream': '24 25 38 49 56 67 115 116 118 140 313 373 491 492 524',
    'savory_crisps': '8 26 27 43 63 104 113 117 149 150 152 155 186 294 494',
    'nuts': '109 110 183 410 416 427 450 451 457 525 539',
    'cheese': '54 70 71 82 92 98 132 175 514 515 516 517 519',
    'eggs': '321 343 560',
    'cooked_sausage_patties': '176 185 190 191 324 496 535 538',
    'breaded_meat': '309 555',
    'roasted_meat': '301 311 312 315 319 328 387 546 552 553 554 557 561 562 563',
    'ham_bacon': '536 548 559',
    'meat_pie': '497 498',
    'raw_beef': '537 540 544',
    'salmon': '307 329 409',
    'fresh_fruit': '192 194 199 200 202 203 204 206 207 209 210 211 214 216 217 218 220 221 222 223 224 230 231 234 235 238 241 243 245 248 254 255 256 269 276 280 281 284 285 325 345 365 386 389 391 392 393 394 396 397 398 399 402 404 407 413 415 436 452 453 454 458 463 464 466 467 468 478 479 528 530 531 534 543',
    'banana': '282 341 379',
    'avocado': '283 542',
    'fresh_nonstarchy_vegetable': '195 197 198 208 233 239 246 247 249 250 251 252 257 258 259 260 262 263 264 265 266 267 268 271 272 273 274 275 277 278 333 334 342 359 360 364 367 368 395 401 405 418 428 429 430 431 432 433 434 435 438 442 444 445 446 448 449 455 460 461 508 520 522 550',
    'cooked_vegetable_side': '288 299 303 356 361 362 370 411 414 447 502 509 529',
    'starchy_vegetable': '270 305 335 346 412 417 424 437 459',
    'pickled_cucumber': '253 380 382 477',
    'milk_coffee': '330 472',
    'beer': '456 518',
    'wine': '388 390',
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def normalize(name):
    s = re.sub(r'[^a-z0-9 ]', ' ', name.lower())
    s = re.sub(r'\s+', ' ', s).strip()
    for a, b in [('chroisant','croissant'),('croissants','croissant'),('rasberries','raspberries'),('rasberry','raspberry'),('icecream','ice cream'),('doughnut','donut')]:
        s = s.replace(a,b)
    return s


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True)
    ap.add_argument('--no-display-length-match', action='store_true')
    ap.add_argument('--preserve-folds-from', help='Reuse prior food groups and folds, never outcome-based repartitioning')
    args=ap.parse_args()
    out=ROOT/args.output
    if out.exists() or not out.resolve().is_relative_to(ROOT/'outputs'):ap.error('Use a new output version')
    out.mkdir(parents=True)
    source=ROOT/'data/prepared/image_agreement_nutrition.csv'
    columns=['image_id','food_name','kcal_100g','category_name','processing','image_path']
    with source.open(encoding='utf-8-sig') as f:
        rows=[{k:r[k] for k in columns} for r in csv.DictReader(f)]
    rows.sort(key=lambda r:int(r['image_id']))
    assigned={}
    for group,ids in GROUPS.items():
        for i in ids.split():
            assert i not in assigned,i
            assigned[i]=group
    grouped=defaultdict(list)
    for r in rows:
        r['pairing_group']=assigned.get(r['image_id'],'unresolved_excluded')
        r['display_kcal_100g']=f"{float(r['kcal_100g']):.1f}"
        r['food_group']=normalize(r['food_name'])
        grouped[r['pairing_group']].append(r)
    seed=20260916
    rng=np.random.default_rng(seed)
    eligible=[];adjacency={};excluded=[]
    for group,rs in sorted(grouped.items()):
        n=len(rs);v=np.array([float(r['display_kcal_100g']) for r in rs])
        lo=np.minimum(v[:,None],v[None,:]);hi=np.maximum(v[:,None],v[None,:])
        allowed=(lo>0)&(hi<=1.5*lo)&(hi>=1.05*lo)&((hi-lo)>=2)
        if not args.no_display_length_match:
            digit=np.array([len(r['display_kcal_100g']) for r in rs])
            allowed &= digit[:,None]==digit[None,:]
        np.fill_diagonal(allowed,False)
        if group=='unresolved_excluded':allowed[:]=False
        cost=np.where(allowed, rng.uniform(0,1,(n,n)),1e9)
        np.fill_diagonal(cost,1e6)
        ri,ci=linear_sum_assignment(cost)
        active=ri[ri!=ci]
        selected=[rs[i] for i in active]
        eligible.extend(r['image_id'] for r in selected)
        adjacency[group]=(selected,allowed[np.ix_(active,active)])
        for i,r in enumerate(rs):
            r['P_eligible']=i in set(active)
            r['pairing_review_status']='assistant_visual_food_form_review_completed_not_human_validation'
            if not r['P_eligible']:
                excluded.append({'image_id':r['image_id'],'food_name':r['food_name'],'pairing_group':group,'reason':'unresolved_food_or_no_complete_plausible_different_value_permutation'})
    permutations=[]
    for k in range(5):
        prng=np.random.default_rng(seed+100+k)
        for group,(rs,allowed) in sorted(adjacency.items()):
            if not rs:continue
            n=len(rs);cost=np.where(allowed,prng.uniform(0,1,(n,n)),1e9)
            ri,ci=linear_sum_assignment(cost)
            assert np.all(allowed[ri,ci]),group
            for i,j in zip(ri,ci):
                permutations.append({'image_id':rs[i]['image_id'],'condition':f'P{k+1}','donor_image_id':rs[j]['image_id'],'pairing_group':group,'true_kcal_100g':rs[i]['display_kcal_100g'],'presented_kcal_100g':rs[j]['display_kcal_100g'],'seed':seed+100+k})
    # Exact-name grouping, with near-duplicate candidates exported for visual review.
    hashes={}
    for r in rows:
        with Image.open(ROOT/r['image_path']) as im:
            arr=np.asarray(im.convert('L').resize((32,32)),dtype=float)
        coeff=dctn(arr,type=2,norm='ortho')[:8,:8].flatten()[1:]
        hashes[r['image_id']]=coeff>np.median(coeff)
    near=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            distance=int(np.sum(hashes[a['image_id']]!=hashes[b['image_id']]))
            if distance<=6 and a['food_group']!=b['food_group']:
                near.append({'image_id_1':a['image_id'],'image_id_2':b['image_id'],'name_1':a['food_name'],'name_2':b['food_name'],'phash_hamming':distance})
    # Visual review rejected the sole <=6 pHash pair (cake100/cookie121).
    # Conservative food-identity groups below address obvious variants/crops.
    parent={r['food_group']:r['food_group'] for r in rows}
    def root(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    byid={r['image_id']:r for r in rows}
    visual_food_groups = [
        '192 204 231 238 404 466', '194 413', '199 452', '200 221 230 235 245 365',
        '206 210 467 531', '202 248', '211 222 234 243 530', '241 396 402',
        '281 284 345 391 394 397 543', '280 398 399', '276 415 458',
        '285 454 478', '427 525', '450 451', '109 183 410',
        '208 334 431', '252 430', '405 508', '195 267', '197 233 275 446 460',
        '246 442', '247 444', '429 445', '198 239', '257 432',
        '249 303 418', '260 359', '265 367', '266 333', '274 455', '435 550',
        '253 380 382 477', '346 412 437', '95 103 420 421', '134 137',
        '403 489', '484 565', '373 492 524', '307 329 409',
        '537 540 544', '96 99 135', '7 15 66 114 120 187 351 443',
        '41 101 119 169 374 375 377 441', '205 213 225 240', '423 425 480',
        '426 440', '439 469', '503 504', '505 506', '338 339',
    ]
    for group in visual_food_groups:
        keys=[root(byid[i]['food_group']) for i in group.split()]
        for k in keys:parent[k]=min(keys)
    fg=defaultdict(list)
    for r in rows:r['food_group']=root(r['food_group']);fg[r['food_group']].append(r)
    order=list(fg);rng.shuffle(order);order.sort(key=lambda g:-sum(r['P_eligible'] for r in fg[g]))
    counts=np.zeros(5,int);allcounts=np.zeros(5,int)
    for g in order:
        fold=min(range(5),key=lambda j:(counts[j],allcounts[j],j))
        for r in fg[g]:r['outer_fold']=fold
        counts[fold]+=sum(r['P_eligible'] for r in fg[g]);allcounts[fold]+=len(fg[g])
    prior_path = ROOT/args.preserve_folds_from if args.preserve_folds_from else None
    if prior_path:
        with prior_path.open() as f:
            prior = {r['image_id']: {k:r[k] for k in ('food_group','outer_fold')}
                     for r in csv.DictReader(f)}
        assert set(prior) == set(byid)
        counts[:]=0; allcounts[:]=0
        for r in rows:
            assert r['food_group'] == prior[r['image_id']]['food_group']
            r['outer_fold']=int(prior[r['image_id']]['outer_fold'])
            counts[r['outer_fold']]+=int(r['P_eligible'])
            allcounts[r['outer_fold']]+=1
    write_csv(out/'stimulus_manifest.csv',rows)
    write_csv(out/'P_permutations.csv',permutations)
    write_csv(out/'excluded_from_P.csv',excluded)
    if near:write_csv(out/'near_duplicate_candidates.csv',near)
    # Contact sheets are for visual pairing review, not scientific results.
    ordered=sorted(rows,key=lambda r:(r['pairing_group'],int(r['image_id'])))
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',13)
    for page,start in enumerate(range(0,len(ordered),64)):
        sheet=Image.new('RGB',(1280,1280),'white');draw=ImageDraw.Draw(sheet)
        for idx,r in enumerate(ordered[start:start+64]):
            x=(idx%8)*160;y=(idx//8)*160
            with Image.open(ROOT/r['image_path']) as im:
                im=im.convert('RGB');im.thumbnail((152,108));sheet.paste(im,(x+(160-im.width)//2,y))
            label=f"{r['image_id']} {r['display_kcal_100g']} {'P' if r['P_eligible'] else 'OUT'}"
            draw.text((x+3,y+110),label,fill='black',font=font)
            draw.text((x+3,y+128),r['food_name'][:23],fill='black',font=font)
            draw.text((x+3,y+144),r['pairing_group'][:22],fill='#555555',font=font)
        sheet.save(out/f'contact_{page+1:02d}.jpg',quality=90)
    for k in range(5):
        ps=[r for r in permutations if r['condition']==f'P{k+1}'];assert len(ps)==len(eligible)
        assert {r['image_id'] for r in ps}=={r['donor_image_id'] for r in ps}==set(eligible)
    report={'created_at':datetime.now().astimezone().isoformat(),'status':'operational_pairing_and_folds_frozen_before_model_effects','images':len(rows),'P_eligible':len(eligible),'excluded':len(excluded),'food_groups':len(fg),'near_duplicate_candidates':len(near),'P_fold_counts':counts.tolist(),'all_image_fold_counts':allcounts.tolist(),'P_seeds':[seed+100+k for k in range(5)],'fold_seed':seed,'pairing_rules':{'same_manually_defined_food_form':True,'ratio_range':[1.05,1.5],'minimum_absolute_difference_kcal':2,'same_display_character_length':True,'display_decimal_places':1,'complete_one_to_one_permutation':True},'provenance':{'input_sha256':sha(source),'script_sha256':sha(__file__),'python':platform.python_version(),'numpy':np.__version__,'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'read_columns':columns},'limitations':['Food-form classes and conservative energy ranges are operational plausibility rules, not human validation','Only image/name/energy/processing metadata read; no rating or neural outcomes used','All contact sheets visually reviewed; sole pHash candidate rejected; explicit conservative identity groups used, not exhaustive near-duplicate guarantee']}
    report['pairing_rules']['same_display_character_length']=not args.no_display_length_match
    report['pairing_rules']['match_token_count']=False
    report['provenance']['arguments']=vars(args)
    report['provenance']['scipy_version']=__import__('scipy').__version__
    report['provenance']['pillow_version']=__import__('PIL').__version__
    report['provenance']['preserved_fold_manifest_sha256']=sha(prior_path) if prior_path else None
    report['fold_assignment']='preserved_from_prior_manifest' if prior_path else 'seeded_greedy_group_balance'
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['images','P_eligible','excluded','food_groups','near_duplicate_candidates','P_fold_counts']}))


if __name__=='__main__':main()
