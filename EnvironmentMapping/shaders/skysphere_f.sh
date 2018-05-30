#version 330

in vec2 texture_coord;
uniform sampler2D texture1;
out vec4 fragColor;

void main (void) {
	vec4 c = texture(texture1, texture_coord);

	fragColor = c;
	fragColor.a = 1.0;
}
