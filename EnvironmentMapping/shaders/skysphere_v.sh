#version 330
in vec3 vertex;
in vec3 normal;
uniform mat4 Projection;
uniform mat4 View;
uniform mat4 Model;
out vec2 texture_coord;

void main() {
	gl_Position = Projection * View * Model * vec4(vertex, 1.0);

	vec3 normal1 = normalize(normal);

	texture_coord = (inverse(transpose(Model)) * vec4(normal1, 0.0)).xy;
}
