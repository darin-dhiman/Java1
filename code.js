var p5Inst = new p5(null, 'sketch');

window.preload = function () {
  initMobileControls(p5Inst);

  p5Inst._predefinedSpriteAnimations = {};
  p5Inst._pauseSpriteAnimationsByDefault = false;
  var animationListJSON = {"orderedKeys":["b013c4db-210f-466c-99c8-b769cf5bb9b1","c475ef93-d8d3-46ab-9b11-fdcdd1fa1408","990377f1-e50c-48ea-a596-02cd8271e556","67f64863-a9e1-4c1d-9567-a303240dac6d","508fb7c0-7018-48a0-8ac0-d77a84c3fbf4","19586d5e-c04f-438e-b783-418fc65c2d33","dc71c34d-a2b3-4a0e-8969-b38487b21610","087790c6-850c-4db5-9ccd-6db048fed169","11d39058-955b-43a3-8e6e-14ff022d1082","1ccd80a5-5a4c-427a-8408-12db4aee2ff9"],"propsByKey":{"b013c4db-210f-466c-99c8-b769cf5bb9b1":{"name":"gnome","sourceUrl":null,"frameSize":{"x":44,"y":136},"frameCount":1,"looping":true,"frameDelay":12,"version":"5ZJlkZg5q7fQIP8pdZyePdnZ3snT.m6A","categories":["fantasy"],"loadedFromSource":true,"saved":true,"sourceSize":{"x":44,"y":136},"rootRelativePath":"assets/b013c4db-210f-466c-99c8-b769cf5bb9b1.png"},"c475ef93-d8d3-46ab-9b11-fdcdd1fa1408":{"name":"vampire","sourceUrl":"assets/api/v1/animation-library/gamelab/IZygDeyZWWN7cOC674xagrMabMM2S.Yf/category_fantasy/monster_19.png","frameSize":{"x":192,"y":358},"frameCount":1,"looping":true,"frameDelay":2,"version":"IZygDeyZWWN7cOC674xagrMabMM2S.Yf","categories":["fantasy"],"loadedFromSource":true,"saved":true,"sourceSize":{"x":192,"y":358},"rootRelativePath":"assets/api/v1/animation-library/gamelab/IZygDeyZWWN7cOC674xagrMabMM2S.Yf/category_fantasy/monster_19.png"},"990377f1-e50c-48ea-a596-02cd8271e556":{"name":"clown","sourceUrl":"assets/api/v1/animation-library/gamelab/5e.55ijwCKCN3A7XzGgPl.3JPFw_78i./category_fantasy/monster_03.png","frameSize":{"x":243,"y":344},"frameCount":1,"looping":true,"frameDelay":2,"version":"5e.55ijwCKCN3A7XzGgPl.3JPFw_78i.","categories":["fantasy"],"loadedFromSource":true,"saved":true,"sourceSize":{"x":243,"y":344},"rootRelativePath":"assets/api/v1/animation-library/gamelab/5e.55ijwCKCN3A7XzGgPl.3JPFw_78i./category_fantasy/monster_03.png"},"67f64863-a9e1-4c1d-9567-a303240dac6d":{"name":"dart","sourceUrl":null,"frameSize":{"x":393,"y":163},"frameCount":1,"looping":true,"frameDelay":12,"version":"fJr41yFseKwdZ2I.pE7affODo98Ggjkb","categories":["sports"],"loadedFromSource":true,"saved":true,"sourceSize":{"x":393,"y":163},"rootRelativePath":"assets/67f64863-a9e1-4c1d-9567-a303240dac6d.png"},"508fb7c0-7018-48a0-8ac0-d77a84c3fbf4":{"name":"it","sourceUrl":"assets/api/v1/animation-library/gamelab/xvxZQmTDRQ2OLsv3piBYg8Ybxxvd3EOw/category_household_objects/curtain_rope.png","frameSize":{"x":40,"y":21},"frameCount":1,"looping":true,"frameDelay":2,"version":"xvxZQmTDRQ2OLsv3piBYg8Ybxxvd3EOw","categories":["household_objects"],"loadedFromSource":true,"saved":true,"sourceSize":{"x":40,"y":21},"rootRelativePath":"assets/api/v1/animation-library/gamelab/xvxZQmTDRQ2OLsv3piBYg8Ybxxvd3EOw/category_household_objects/curtain_rope.png"},"19586d5e-c04f-438e-b783-418fc65c2d33":{"name":"coin_gold_1","sourceUrl":"assets/api/v1/animation-library/gamelab/pUFPchUgZRoy5C6YtEEkLfSDmZWPxW.y/category_board_games_and_cards/coin_gold.png","frameSize":{"x":61,"y":61},"frameCount":1,"looping":true,"frameDelay":2,"version":"pUFPchUgZRoy5C6YtEEkLfSDmZWPxW.y","categories":["board_games_and_cards"],"loadedFromSource":true,"saved":true,"sourceSize":{"x":61,"y":61},"rootRelativePath":"assets/api/v1/animation-library/gamelab/pUFPchUgZRoy5C6YtEEkLfSDmZWPxW.y/category_board_games_and_cards/coin_gold.png"},"dc71c34d-a2b3-4a0e-8969-b38487b21610":{"name":"knife_1","sourceUrl":null,"frameSize":{"x":19,"y":105},"frameCount":1,"looping":true,"frameDelay":12,"version":"J.JZsIagWFh9vqd3uDDOYj_9_voULDqJ","categories":["tools"],"loadedFromSource":true,"saved":true,"sourceSize":{"x":19,"y":105},"rootRelativePath":"assets/dc71c34d-a2b3-4a0e-8969-b38487b21610.png"},"087790c6-850c-4db5-9ccd-6db048fed169":{"name":"farm_land_1","sourceUrl":null,"frameSize":{"x":400,"y":400},"frameCount":1,"looping":true,"frameDelay":12,"version":"UHnWsCzQgIG9VudjPP1U9hob2y5HI7jz","categories":["backgrounds"],"loadedFromSource":true,"saved":true,"sourceSize":{"x":400,"y":400},"rootRelativePath":"assets/087790c6-850c-4db5-9ccd-6db048fed169.png"},"11d39058-955b-43a3-8e6e-14ff022d1082":{"name":"lost","sourceUrl":null,"frameSize":{"x":339,"y":190},"frameCount":1,"looping":true,"frameDelay":12,"version":"iylFmduOZBAsMegStRx3ULeCwsGJUvIj","categories":[""],"loadedFromSource":true,"saved":true,"sourceSize":{"x":339,"y":190},"rootRelativePath":"assets/11d39058-955b-43a3-8e6e-14ff022d1082.png"},"1ccd80a5-5a4c-427a-8408-12db4aee2ff9":{"name":"won","sourceUrl":"assets/v3/animations/yzYO011Lp48ZIlxhyxKvfklTFtd5i0-oA3NwGA-le7c/1ccd80a5-5a4c-427a-8408-12db4aee2ff9.png","frameSize":{"x":268,"y":191},"frameCount":1,"looping":true,"frameDelay":4,"version":"yDg7Fm7NnCKAkxQ8sxww4GChcZfx4Fv5","categories":[""],"loadedFromSource":true,"saved":true,"sourceSize":{"x":268,"y":191},"rootRelativePath":"assets/v3/animations/yzYO011Lp48ZIlxhyxKvfklTFtd5i0-oA3NwGA-le7c/1ccd80a5-5a4c-427a-8408-12db4aee2ff9.png"}}};
  var orderedKeys = animationListJSON.orderedKeys;
  var allAnimationsSingleFrame = false;
  orderedKeys.forEach(function (key) {
    var props = animationListJSON.propsByKey[key];
    var frameCount = allAnimationsSingleFrame ? 1 : props.frameCount;
    var image = loadImage(props.rootRelativePath, function () {
      var spriteSheet = loadSpriteSheet(
          image,
          props.frameSize.x,
          props.frameSize.y,
          frameCount
      );
      p5Inst._predefinedSpriteAnimations[props.name] = loadAnimation(spriteSheet);
      p5Inst._predefinedSpriteAnimations[props.name].looping = props.looping;
      p5Inst._predefinedSpriteAnimations[props.name].frameDelay = props.frameDelay;
    });
  });

  function wrappedExportedCode(stage) {
    if (stage === 'preload') {
      if (setup !== window.setup) {
        window.setup = setup;
      } else {
        return;
      }
    }
// -----

// Create your variables here
var Points = 0;
var WellBeing = 10;
// Create your sprites here
var farm = createSprite(200, 200);
farm.setAnimation("farm_land_1");

var gnome = createSprite(200, 300);
gnome.setAnimation("gnome");
var vampire = createSprite(100,100);
vampire.setAnimation("vampire");
vampire.height = 120;
vampire.width = 100;
vampire.visible = false;
var clown = createSprite(randomNumber(75, 325), -75);
clown.setAnimation("clown");
clown.height = 120;
clown.width = 100;
var gold = createSprite(randomNumber(75, 325), 75);
gold.setAnimation("coin_gold_1");
var knife1 = createSprite(randomNumber(75, 325), 0);
knife1.setAnimation("knife_1");
knife1.velocityY = randomNumber(1,4);
var knife2 = createSprite(randomNumber(75, 325), 0);
knife2.setAnimation("knife_1");
knife2.velocityY = randomNumber(3, 6);


function main(){

  gnome.visible = true;
  clown.visible = true;
  clown.velocityY = randomNumber(1,4);
  gold.velocityY = 2;
  
  background("farm-land1");

  
  if(keyDown("left")){
    gnome.velocityX = -3;
  }

  if(keyDown("right")){
    gnome.velocityX = 3;
  }   

  //if (clown.velocityX < 0){
  //}
  if (clown.x < 0){
    clown.x = randomNumber(75, 325);
    clown.y = 75;
  }
  if (clown.x > 400){
    clown.x = randomNumber(75, 325);
    clown.y = 75;
  }
  
  if (gnome.isTouching(clown)){
    clown.velocityY = 0;
    gnome.displace(clown);
  }
  if (clown.y > 200){
   if (Points > 0){
    Points-=1;
    clown.x = randomNumber(75, 325);
    clown.y = 75;
    
   }
  }
  if (gold.x < 0){
    gold.x = randomNumber(75, 325);
    gold.y = 75;
  }
  if (gold.x > 400){
    gold.x = randomNumber(75, 325);
    gold.y = 75;
  }
  if (gold.y > 400){
    gold.x = randomNumber(75, 325);
    gold.y = 75;
  }
  if (gold.isTouching(gnome)){
   Points = Points+1;
   gold.x = randomNumber(75, 325);
   gold.y = 75;
   }
  if (knife1.isTouching(gnome)){
    WellBeing = WellBeing-1;
   }
  if (knife2.isTouching(gnome)){
    WellBeing = WellBeing-1;
   }
  if (knife1.isTouching(clown)){
    clown.x = randomNumber(75, 325);
    clown.y = 75;  
   }
  if (knife2.isTouching(clown)){
    clown.x = randomNumber(75, 325);
    clown.y = 75;
   }
  if (knife1.y > 400){
    knife1.x = randomNumber(15, 385);
    knife1.y = randomNumber(-10, 50);
    knife1.velocityY = randomNumber(1, 4);
   }
  if (knife2.y > 400){
    knife2.x = randomNumber(15, 385);
    knife2.y = randomNumber(-10, 50);
    knife2.velocityY = randomNumber(3, 6);
   }
  if(WellBeing < 1){
    WellBeing = 0;
      }
  if(Points > 10){
    Points = 10;
  }

}

function draw() {
// draw background


// update sprites
function wonit(){
  gnome.visible = false;
  vampire.visible = false;
  clown.visible = false;
  gold.visible = false;
  knife1.visible = false;
  knife2.visible = false;
  fill("black");
  textSize(40);
  text("  You Won!!!!!", 10, 200);
  
  
  //if(keyDown(!"space")){

  }
  function lostit(){
    
    gnome.visible = false;
    vampire.visible = false;
    clown.visible = false;
    gold.visible = false;
    knife1.visible = false;
    knife2.visible = false;
    clown.setAnimation("gnome");
    background("blue");
    fill("black");
    textSize(40);
    text("  You lost the game.", 10, 200);
    var lost = createSprite(200, 200);
    lost.setAnimation("lost");
  }
  // Create your functions here
  
    if (WellBeing < 1 && Points < 10){
      lostit();
      fill("black");
      textSize(40);
      var lost = createSprite(200, 200);
      lost.setAnimation("lost");
      lost.scale = 1;
      lost.height = 400;
      lost.width = 400;

      gold.velocity = 0;

    }
    if (Points > 9 ){
      wonit();
      var won = createSprite(200, 200);
      won.setAnimation("won");
      won.scale = 1;
      won.height = 400;
      won.width = 400;
      clown.velocityY = 0;

      if (Points > Points){
        Points -= 1;
      }
      if (Points -= 1){
        Points += 1;
      }

    } else{
      main();
    }
    
    
  
  
   // gnome.visible = true;
   // vampire.visible = true;
   // clown.visible = true;
      drawSprites();
      fill("black");
      textSize(30);
      text("Points:", 20, 30);
      text(Points, 120, 30);
      text("Health:", 240, 30);
      text(WellBeing, 340, 30);

  }




// -----
    try { window.draw = draw; } catch (e) {}
    switch (stage) {
      case 'preload':
        if (preload !== window.preload) { preload(); }
        break;
      case 'setup':
        if (setup !== window.setup) { setup(); }
        break;
    }
  }
  window.wrappedExportedCode = wrappedExportedCode;
  wrappedExportedCode('preload');
};

window.setup = function () {
  window.wrappedExportedCode('setup');
};
