let state = {
  'svg': null,
  'lambda': null,
  'phi': null,
  'projection': null,
  'lines': null,
  't': null,
  'pitch': 0,
  'transformx': 0,
  'transformy': 0, 
  'G': G,
  'points': null,
  'edges': null
}


function rad2deg(r){
  return r * 180/Math.PI;
}
function rad2deg_arr(arr){
  return [rad2deg( arr[0] ), rad2deg( arr[1] )]
}
function deg2rad(d){
  return d * Math.PI/180;
}

function inv_stereo(x,y){
  x2 = x*x;
  y2 = y*y;
  newX = (2*x)/(1+x2+y2);
  newY = (2*y)/(1+x2+y2);
  z = (x2+y2-1)/(1+x2+y2);
  return [newX,newY,z];
}

function cartToPolar(v){
  x = v[0];
  y = v[1];
  z = v[2];

  r = Math.sqrt(x*x+y*y+z*z);
  x1 = (x/r);
  y1 = (y/r);
  z1 = (z/r);

  polar = Math.acos(y1/200)
  azimuthal = Math.atan2(x1,z1)
  return [polar,azimuthal]
}

function placeOnSphere(cartCoord){
  cartCoord = parse_pos(cartCoord);
  x = cartCoord[0];
  y = cartCoord[1];
  z = cartCoord[2];

  //v = inv_stereo(x,y)
  //v = cartToPolar([x,y,z])
  v = new THREE.Vector3(x,y,z);

  return vector3toLonLat(v)
}

function parse_pos_2(coord){
  Coords = coord.split(',');
  return([parseFloat(Coords[0]),parseFloat(Coords[1])]);
}

function placeOnSphereNew(sphereCoord){
  sphereCoord = parse_pos_2(sphereCoord);


  return [ sphereCoord[0]*(180.0/Math.PI),sphereCoord[1]*(180.0/Math.PI) ]
}



var width = 1000,
  height = 1000,
  scale = 100,
  lastX = 0,
  lastY = 0,
  origin = {
    x: 0,
    y: 0
  };

//console.log(G)

//var Graph = DotParser.parse(readTextFile('new_outputs/cube_animation0.dot'));

let assign_pos = function(){
  points = {
    type: "FeatureCollection",
    features: G['nodes'].map((val, i) => {
      return {type:"Feature",
              geometry: {
                type: "Point",
                coordinates: rad2deg_arr(val['pos']),
                label: val['id'],
                'class': val.class
              }
        }
    })
  }
  
  
  index_map = {}
  for (const ind in G.nodes) {
    index_map[G.nodes[ind].id] = Number(ind)
  }
  
  console.log(index_map)
  
  edges = G['edges'].map( (val,i) => {
      return {type: "LineString", coordinates: [ rad2deg_arr( G.nodes[index_map[val.source]].pos ), rad2deg_arr( G.nodes[index_map[val.target]].pos ) ] }
    } )
  
  console.log(edges)
  
}


//Map = makeMap(myGraph);

window.addEventListener('load',function(){
  var svg = d3.select("svg");

  //setup_graph_select();
  assign_pos();

  var projection = d3.geoOrthographic(),
      path = d3.geoPath().projection(projection)
      //.pointRadius(d => 1)
      console.log(path)
        //.attr('transform', 0);
  state.projection = projection
  state.path = path
  state.projection.rotate([0, 0,90])


  // zoom AND rotate
   svg.call(d3.zoom().on('zoom', zoomed));

   // code snippet from http://stackoverflow.com/questions/36614251
   state.lambda = d3.scaleLinear()
     .domain([-width, width])
     .range([-180, 180])

   state.phi = d3.scaleLinear()
     .domain([-height, height])
     .range([90, -90]);

  state.clr_map = ["#179d17","#68dca4","#f21011","#7eefda","#0302f3","#b87676","#ea86e8","#f8b22d","#e9e889","#b53e3e", "#a316fb", "#e6855b", "#e9ea1f"]

  let s_height = document.querySelector('#div1').offsetHeight /2
  let s_width = document.querySelector('#div1').offsetWidth /2

  svg.append('path')
      .attr('id', 'sphere')
      .datum({ type: "Sphere" })
      .attr('d', path);

  state.svg = svg

  state.lines = d3.geoGraticule().step([10, 10]);
  state.t = d3.transition().duration(750)

  updateData()
})

let setup_graph_select = function(){
  var values = G_names;
  values.sort()
  
  var select = document.createElement("select");
  select.name = "pets";
  select.id = "pets"

  var option = document.createElement("option");
  option.value = null;
  option.text = "Choose a graph";
  select.appendChild(option);

  for (const val of values)
  {
      var option = document.createElement("option");
      option.value = val;
      option.text = val;
      select.appendChild(option);
  }

  var label = document.createElement("label");
  // label.innerHTML = "Choose your pets: "
  // label.htmlFor = "pets";

  select.addEventListener('change', d =>{
    var script = document.createElement('script');
      script.onload = function () {
          state.G = G
          assign_pos();
          state.svg.selectAll(".links").remove()
          state.svg.selectAll(".sites").remove()
          state.svg.selectAll(".graticule").remove()

          updateData();
        };
    script.src = "exp_drawings/" + select.value + ".js";
    document.head.appendChild(script);
  })

  document.getElementById("selection").appendChild(label).appendChild(select);
}

let sliderchange = function(){
  state.pitch = document.getElementById("pitch").value;
  state.projection.rotate([state.transformx,state.transformy,state.pitch])
  updatePaths()
}


function updateData(){

  let svg = state.svg
  let lines = state.lines
  let path = state.path
  let t = state.t

  svg.append('g')
      .attr('class', 'graticule')
      .selectAll('path')
      .data([lines()])
      .enter()
      .append('path')
      .attr('d', path)


  let mylinks = svg.append('g')
      .attr('class', 'links')
      .selectAll('path')
      .data(edges)
      .enter()
      .append('path')
      .attr('d', path);


  svg.append('g')
      .attr('class', 'sites')
      .selectAll('path')
      .data(points.features)
      .join(
        enter => enter.append('path')
        .attr('d', path)
        .attr('fill', d => {
          console.log(d)
          return d.geometry.class ? state.clr_map[d.geometry.class] : null
        }),

        update => update.transition(t)
          .attr('d',path)
      );


  let labels = svg.append('g').attr('class', 'labels');

  // svg.append('g')
  //     .attr('class', 'labels')
  //     .selectAll('path')
  //     .data(points.features)
  //     .enter()
  //     .append('text')
  //     .text(d => d.label)
  //     //.attr('d',d => d.label)
  //     .attr('font-size', 7)
  //     .style('text-anchor', 'middle')
  //     .attr('transform', function(d) {
  //          return 'translate(' +  path.centroid(d) + ')';
  //        })
  //        .text(function(d) {return d.geometry.label; });
  }


  function zoomed() {
    var transform = d3.event.transform;
    var r = {
      x: state.lambda(transform.x),
      y: state.phi(transform.y)
    };
    var k = Math.sqrt(100 / state.projection.scale());
    if (d3.event.sourceEvent.wheelDelta) {
      state.projection.scale(10 * transform.k)

    } else {
      state.projection.rotate([origin.x + r.x, origin.y + r.y, state.pitch]);
      state.transformx = transform.x;
      state.transformy = transform.y;
    }
    updatePaths();
  }

function updatePaths(){
  let svg = state.svg
  let path = state.path

  svg.selectAll('path')
    .attr('d',path);

    //Remove and redraw text
    d3.selectAll('text').remove()

    // svg.append('g')
    //     .attr('class', 'labels')
    //     .selectAll('path')
    //     .data(points.features)
    //     .enter()
    //     .append('text')
    //     .text(d => d.label)
    //     //.attr('d',d => d.label)
    //     .attr('font-size', 10)
    //     .style('text-anchor', 'middle')
    //     .attr('transform', function(d) {
    //          return 'translate(' +  path.centroid(d) + ')';
    //        })
    //        .text(function(d) {return d.geometry.label; });

}

function changeProjection(){
  switch(document.getElementById('projection').value){
    case "stereographic":
      state.projection = d3.geoStereographic(),
          state.path = d3.geoPath().projection(state.projection);
      break;

    case "mercator":
      state.projection = d3.geoMercator(),
          state.path = d3.geoPath().projection(state.projection);
      break;

    case "equalearth":
      state.projection = d3.geoEqualEarth(),
          state.path = d3.geoPath().projection(state.projection)//.pointRadius(d => 1);
      break;

    default:
      state.projection = d3.geoOrthographic(),
          state.path = d3.geoPath().projection(state.projection);
  }
  updatePaths();
}


// gentle animation
// d3.interval(function(elapsed) {
//     projection.rotate([ elapsed / 150, 0 ]);
//     svg.selectAll('path')
//         .attr('d', path);
// }, 50);

//updateData(points)
